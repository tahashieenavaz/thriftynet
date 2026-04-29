import torch
from typing import Type


class ThriftyEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        filters: int,
        iterations: int,
        kernel_size: int,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()
        self.filters = filters
        self.iterations = iterations
        self.conv = torch.nn.Conv2d(
            filters,
            filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.activation = activation()
        self.normalizations = torch.nn.ModuleList(
            [torch.nn.BatchNorm2d(filters) for _ in range(iterations)]
        )
        alpha = torch.zeros(iterations, 2)
        alpha[:, 0] = 0.1
        alpha[:, 1] = 0.9
        self.alpha = torch.nn.Parameter(alpha)

        pool_every = max(1, iterations // 5)
        self.pool_strategy = [
            t % pool_every == pool_every - 1 for t in range(iterations)
        ]
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = torch.nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.filters > x.size(1):
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.filters - x.size(1)))

        for t in range(self.iterations):
            a_t = self.activation(self.conv(x))
            x = self.alpha[t, 0] * a_t + self.alpha[t, 1] * x
            x = self.normalizations[t](x)
            if self.pool_strategy[t]:
                x = self.pool(x)

        x = self.global_pool(x)
        return x.flatten(1)


if __name__ == "__main__":
    encoder = ThriftyEncoder(filters=128, iterations=20, kernel_size=3)
    images = torch.randn(1, 3, 84, 84)
    features = encoder(images)
    assert features.size(1) == 128

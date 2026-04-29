import torch
from typing import Type
from typing import Literal


class ThriftyEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        filters: int,
        iterations: int,
        kernel_size: int,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
        normalization: (
            Type[torch.nn.Module] | Literal["batch", "layer", "group"]
        ) = "batch",
    ):
        super().__init__()
        self.filters = filters
        self.iterations = iterations
        self.__initialize_normalizations(normalization=normalization)

        self.conv = torch.nn.Conv2d(
            filters,
            filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.activation = activation()
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

    def __initialize_normalizations(
        self, normalization: Type[torch.nn.Module] | Literal["batch", "layer", "group"]
    ):
        normalization_map = {
            "group": lambda: torch.nn.GroupNorm(self.filters // 16, self.filters),
            "layer": lambda: torch.nn.GroupNorm(1, self.filters),
            "batch": lambda: torch.nn.BatchNorm2d(self.filters),
        }

        if isinstance(normalization, torch.nn.Module):
            normalization_fn = lambda: normalization(self.filters)
        else:
            normalization_fn = normalization_map[normalization]

        self.normalizations = torch.nn.ModuleList(
            [normalization_fn() for _ in range(self.iterations)]
        )

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

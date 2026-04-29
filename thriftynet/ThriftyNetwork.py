import torch
from typing import Type
from .ThriftyEncoder import ThriftyEncoder


class ThriftyNetwork(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        filters: int = 128,
        iterations: int = 20,
        kernel_size: int = 3,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()
        self.encoder = ThriftyEncoder(
            filters=filters,
            iterations=iterations,
            activation=activation,
            kernel_size=kernel_size,
        )
        self.head = torch.nn.Linear(filters, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.head(features)


if __name__ == "__main__":
    encoder = ThriftyNetwork(10, filters=128, iterations=20, kernel_size=3)
    images = torch.randn(1, 3, 84, 84)
    features = encoder(images)
    assert features.size(1) == 10

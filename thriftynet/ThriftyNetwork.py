import torch
from typing import Type
from .ThriftyEncoder import ThriftyEncoder


class ThriftyNetwork(torch.nn.Module):
    def __init__(
        self,
        classes: int,
        *,
        filters: int = 128,
        iterations: int = 20,
        activation: Type[torch.nn.Module] = torch.nn.ReLU,
    ):
        self.encoder = ThriftyEncoder(
            filters=filters, iterations=iterations, activation=activation
        )
        self.head = torch.nn.Linear(filters, classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.head(features)

from __future__ import annotations

import torch
import torch.nn as nn


class MLPForecaster(nn.Module):
    """Simple MLP: context window -> multi-step horizon forecast."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.net = nn.Sequential(
            nn.Linear(context_length, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

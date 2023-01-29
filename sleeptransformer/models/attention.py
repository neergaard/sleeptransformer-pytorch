import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_dimension: int, attention_dimension: int) -> None:
        super().__init__()
        self.input_dimension = input_dimension
        self.attention_dimension = attention_dimension

        self.a_transform = nn.Sequential(
            nn.Linear(in_features=self.input_dimension, out_features=self.attention_dimension, bias=True), nn.Tanh()
        )
        self.attention_transform = nn.Sequential(
            nn.Linear(in_features=self.attention_dimension, out_features=1, bias=False),
            nn.Flatten(),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        a_t = self.a_transform(x)
        attention_weights = self.attention_transform(a_t)
        context_vector = (attention_weights.unsqueeze(2) * x).sum(dim=1)

        return context_vector, attention_weights

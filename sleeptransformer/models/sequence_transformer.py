import torch
import torch.nn as nn

from .base_transformer import BaseTransformer


class SequenceTransformer(BaseTransformer):
    def __init__(
        self,
        fc_dim: int,
        n_heads: int,
        dropout: float,
        n_layers: int,
        input_dim: int,
        n_classes: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(n_heads=n_heads, dropout=dropout, n_layers=n_layers, input_dim=input_dim, hidden_dim=hidden_dim)
        self.fc_dim = fc_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim

        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=fc_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(in_features=fc_dim, out_features=n_classes),
            nn.Dropout(p=dropout),
            nn.ReLU(),
        )

    def forward(self, X: torch.Tensor) -> None:

        # Pass the sequences through the Sequence Transformer
        z, att_weights = super().forward(X)

        # Pass the outputs through the linear layers
        y = self.linear_layer(z)

        return y, att_weights

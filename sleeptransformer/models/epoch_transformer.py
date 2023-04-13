from typing import List, Tuple
import torch

from .attention import AttentionLayer
from .base_transformer import BaseTransformer


class EpochTransformer(BaseTransformer):
    def __init__(self, n_heads: int, dropout: float, n_layers: int, input_dim: int, hidden_dim: int, attention_dim: int) -> None:
        super().__init__(n_heads=n_heads, dropout=dropout, n_layers=n_layers, input_dim=input_dim, hidden_dim=hidden_dim)
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim

        self.attention_embedding = AttentionLayer(input_dimension=input_dim, attention_dimension=attention_dim)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:

        z, att_weights = super().forward(X)

        # Run forward pass through attention layer
        z, alpha = self.attention_embedding(z)

        return z, att_weights, alpha

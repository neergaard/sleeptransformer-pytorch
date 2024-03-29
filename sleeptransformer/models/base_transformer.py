import torch
import torch.nn as nn
from pytorch_lightning.core.mixins import HyperparametersMixin


class BaseTransformer(HyperparametersMixin, nn.Module):
    def __init__(
        self,
        n_heads: int,
        dropout: float,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        encoding_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoding_layer, num_layers=n_layers)
        self.attention_weights_saver = SaveOutput()
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                self.patch_attention(module)
                module.register_forward_hook(self.attention_weights_saver)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, F = x.shape

        # Positional encoding
        z = self._positional_embedding(x)

        # run forward pass through transformer
        z = self.transformer_encoder(z)

        att_weights = torch.stack(self.attention_weights_saver.outputs)
        self.attention_weights_saver.clear()

        return z, att_weights

    def _positional_embedding(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.zeros_like(x).sum(1, keepdim=True)
        T, _, F = x.shape
        for j in torch.arange(0, F, dtype=int):
            if j % 2 == 0:
                p[:, 0, j] = torch.sin(torch.arange(0, T, dtype=int) / 10_000 ** (2 * j / F))
            else:
                p[:, 0, j] = torch.cos(torch.arange(0, T, dtype=int) / 10_000 ** (2 * j / F))

        return x + p.expand_as(x)

    @staticmethod
    def patch_attention(m):
        forward_orig = m.forward

        def wrap(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True

            return forward_orig(*args, **kwargs)

        m.forward = wrap


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

import torch

from sleeptransformer.models.sequence_transformer import SequenceTransformer


class TestSequenceTransformer:
    def test_instance(self) -> None:
        sequence_transformer = SequenceTransformer(
            fc_dim=1024, n_heads=8, dropout=0.1, n_layers=4, input_dim=128, hidden_dim=1024, n_classes=5
        )

        assert sequence_transformer

    def test_forward(self) -> None:
        N, T, F, K = 32, 21, 128, 5
        sequence_transformer = SequenceTransformer(
            fc_dim=1024, n_heads=8, dropout=0.1, n_layers=4, input_dim=128, hidden_dim=1024, n_classes=5
        )
        X = torch.rand((N, T, F))

        y = sequence_transformer(X)

        assert y.shape == (N, T, K)

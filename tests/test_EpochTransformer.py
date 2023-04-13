import torch

from sleeptransformer.models.epoch_transformer import EpochTransformer


class TestEpochTransformer:
    def test_instance(self) -> None:
        epoch_transformer = EpochTransformer(8, 0.1, 4, 128, 1024, 1024)

        assert epoch_transformer

    def test_forward(self) -> None:
        N, T, F = 32, 29, 128
        epoch_transformer = EpochTransformer(8, 0.1, 4, 128, 1024, 1024)
        X = torch.rand((N, T, F))

        z, att_weights, alpha = epoch_transformer(X)

        assert z.shape == (N, F)
        for a in att_weights:
            assert a.shape == (N, T, T)
        assert alpha.shape == (N, T)

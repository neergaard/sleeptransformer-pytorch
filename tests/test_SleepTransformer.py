import torch

from sleeptransformer.models import SleepTransformer
from sleeptransformer.models.epoch_transformer import EpochTransformer
from sleeptransformer.models.sequence_transformer import SequenceTransformer


class TestSleepTransformer:

    N, L, T, F, K = 32, 21, 29, 128, 5
    epoch_transformer = EpochTransformer(
        n_heads=8, dropout=0.1, n_layers=4, input_dim=F, hidden_dim=1024, attention_dim=128
    )
    sequence_transformer = SequenceTransformer(
        fc_dim=1024, n_heads=8, dropout=0.1, n_layers=4, input_dim=F, hidden_dim=1024, n_classes=K
    )

    def test_instance(self):

        sleep_transformer = SleepTransformer(
            epoch_transformer=self.epoch_transformer, sequence_transformer=self.sequence_transformer
        )
        assert sleep_transformer

    def test_forward(self):
        sleep_transformer = SleepTransformer(
            epoch_transformer=self.epoch_transformer, sequence_transformer=self.sequence_transformer
        )
        X = torch.randn((self.N, self.L, self.T, self.F))

        y, alpha = sleep_transformer(X)

        assert y.shape == (self.N, self.L, self.K)
        assert alpha.shape == (self.N, self.L, self.T)

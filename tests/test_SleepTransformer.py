import torch

from sleeptransformer.loss import SequenceLoss
from sleeptransformer.models import SleepTransformer
from sleeptransformer.models.epoch_transformer import EpochTransformer
from sleeptransformer.models.sequence_transformer import SequenceTransformer


class TestSleepTransformer:

    N, L, T, F, K = 32, 21, 29, 128, 5
    epoch_transformer = EpochTransformer(n_heads=8, dropout=0.1, n_layers=4, input_dim=F, hidden_dim=1024, attention_dim=128)
    sequence_transformer = SequenceTransformer(
        fc_dim=1024, n_heads=8, dropout=0.1, n_layers=4, input_dim=F, hidden_dim=1024, n_classes=K
    )
    sleep_transformer = SleepTransformer(
        epoch_transformer=epoch_transformer,
        sequence_transformer=sequence_transformer,
        loss_fn=SequenceLoss,
        optimizer_params=dict(lr=1e-4),
    )

    def test_instance(self):
        assert self.sleep_transformer

    def test_forward(self):
        X = torch.randn((self.N, self.L, self.T, self.F))

        y, epoch_att_weights, seq_att_weights, alpha = self.sleep_transformer(X)

        assert y.shape == (self.N, self.L, self.K)
        assert alpha.shape == (self.N, self.L, self.T)

    def test_confidence(self):
        N, L, K = 32, 21, 5
        y_pred = torch.rand((N, L, K))
        conf = self.sleep_transformer.compute_confidence(y_pred)
        assert conf.shape == (N, L)

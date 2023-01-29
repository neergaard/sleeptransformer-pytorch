import torch

from sleeptransformer.loss import SequenceLoss


class TestSequenceLoss:

    sequence_loss = SequenceLoss()

    def test_instance(self) -> None:
        assert self.sequence_loss

    def test_forward(self) -> None:
        N, L, K = 32, 21, 5
        y_predict = torch.rand((N, L, K))
        y_target = torch.argmax(torch.randn((N, L, K)), dim=-1)

        loss = self.sequence_loss(y_predict, y_target)

        assert loss.shape == (N,)

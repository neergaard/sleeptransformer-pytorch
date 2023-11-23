from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

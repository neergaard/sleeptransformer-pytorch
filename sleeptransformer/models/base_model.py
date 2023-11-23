from pytorch_lightning import LightningModule


class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()

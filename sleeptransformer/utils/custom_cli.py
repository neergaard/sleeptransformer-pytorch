from datetime import datetime
from pathlib import Path

from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class CustomCLI(LightningCLI):
    """
    Custom LightningCLI handler to handle linking arguments between model components.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--name", default="test", type=str)
        parser.add_argument("--project", default="sleeptransformer", type=str)

    def before_instantiate_classes(self) -> None:
        ...
        # self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self) -> None:
        subcommand = getattr(self.config, "subcommand", None)
        if subcommand is not None:
            name = Path(self.config[subcommand]["name"])
            # project = self.config[subcommand]["project"]
            trainer = self.config[subcommand]["trainer"]
            run_dir = Path(trainer["default_root_dir"])

            if subcommand == "fit":
                ID = wandb.util.generate_id()
                CURRENT_TIME = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
                if name is None:
                    name = CURRENT_TIME / ID
                else:
                    name = name / CURRENT_TIME / ID

                run_dir = run_dir / name
                run_dir.mkdir(parents=True, exist_ok=True)
                run_dir = str(run_dir)
                name = str(name)

                self.config[subcommand]["trainer"]["default_root_dir"] = run_dir

                # Find model checkpoint callback
                callback_idx = [
                    idx
                    for idx, callback in enumerate(self.config[subcommand]["trainer"]["callbacks"])
                    if "ModelCheckpoint" in callback["class_path"]
                ][0]
                self.config[subcommand]["trainer"]["callbacks"][callback_idx]["init_args"]["dirpath"] = run_dir
                if self.config[subcommand]["trainer"]["logger"]:
                    self.config[subcommand]["trainer"]["logger"]["init_args"]["name"] = name
                    self.config[subcommand]["trainer"]["logger"]["init_args"]["dir"] = run_dir

                # Other cli args
                # n_encoder_layers = self.config[subcommand]["n_encoder_layers"]
                # print(n_encoder_layers)
                # if n_encoder_layers is not None:
                #     self.config[subcommand]["model"]["init_args"]["style_encoder"]["init_args"][
                #         "n_encoder_layers"
                #     ] = n_encoder_layers

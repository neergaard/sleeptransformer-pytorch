import warnings

from sleeptransformer.data.base_datamodule import BaseDataModule
from sleeptransformer.models.base_model import BaseModel
from sleeptransformer.utils.custom_cli import CustomCLI
from sleeptransformer.utils.logger import get_logger

warnings.filterwarnings("ignore", ".*does not have many workers.*")
logger = get_logger()


if __name__ == "__main__":
    cli = CustomCLI(
        BaseModel,
        BaseDataModule,
        seed_everything_default=1337,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={
            "parser_mode": "omegaconf",
            "error_handler": None,
            "fit": {"default_config_files": ["sleeptransformer-pytorch/config/sleeptransformer.yaml"]},
        },
    )
    if cli.subcommand == "fit":
        logger.info("Model fitting complete.")
        logger.info("Running test routine...")
        cli.trainer.test(dataloaders=cli.datamodule.test_dataloader())
    if cli.subcommand in ["fit", "test"]:
        logger.info("Test routine finished.")

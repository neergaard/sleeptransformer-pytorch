import os
import subprocess

from sleeptransformer.utils.logger import get_logger

logger = get_logger()


def download_wsc(out_dataset_folder: str, *args, **kwargs):
    """Download the WSC dataset from NSRR.
    This function will download the standardized NSRR .xml files and the EDFs,
    and also the dataset files containing metadata, demographics, etc.
    """
    logger.info("==================================")
    logger.info("DOWNLOADING WSC DATASET")
    logger.info("----------------------------------")

    # Make directory
    if not os.path.exists(out_dataset_folder):
        logger.info(f"Creating output directory {out_dataset_folder}")
        os.makedirs(out_dataset_folder)

    # Use the NSRR Ruby gem to download the dataset
    current_dir = os.getcwd()
    logger.info(f"Changing directory to {out_dataset_folder}")
    os.chdir(out_dataset_folder)
    logger.info("Downloading documentation")
    subprocess.run(["nsrr", "download", "wsc/documentation", "--fast"])
    logger.info("Downloading metadata and demographics")
    subprocess.run(["nsrr", "download", "wsc/datasets", "--fast"])
    logger.info("Downloading annotations and EDFs")
    subprocess.run(["nsrr", "download", "wsc/polysomnography", "--fast"])
    logger.info(f"Changing directory back to {current_dir}")
    os.chdir(current_dir)
    logger.info("Download complete!")

import os
import subprocess

from sleeptransformer.utils.logger import get_logger

logger = get_logger()


def download_shhs(out_dataset_folder: str, *args, **kwargs):
    """Download the SHHS dataset from NSRR.
    This function will download the standardized NSRR .xml files and the EDFs from both subsets,
    and also the dataset files containing metadata, demographics, etc.
    """
    logger.info("==================================")
    logger.info("DOWNLOADING SHHS DATASET")
    logger.info("----------------------------------")

    # Make directory
    if not os.path.exists(out_dataset_folder):
        logger.info(f"Creating output directory {out_dataset_folder}")
        os.makedirs(out_dataset_folder)

    # Use the NSRR Ruby gem to download the dataset
    current_dir = os.getcwd()
    logger.info(f"Changing directory to {out_dataset_folder}")
    os.chdir(out_dataset_folder)
    logger.info("Downloading metadata and demographics")
    subprocess.run(["nsrr", "download", "shhs/datasets", "--fast"])
    logger.info("Downloading documentation")
    subprocess.run(["nsrr", "download", "shhs/documentation", "--fast"])
    logger.info("Downloading annotations from subset 1")
    subprocess.run(["nsrr", "download", "shhs/polysomnography/annotations-events-nsrr/shhs1", "--fast"])
    logger.info("Downloading annotations from subset 2")
    subprocess.run(["nsrr", "download", "shhs/polysomnography/annotations-events-nsrr/shhs2", "--fast"])
    logger.info("Downloading studies from subset 1")
    subprocess.run(["nsrr", "download", "shhs/polysomnography/edfs/shhs1", "--fast"])
    logger.info("Downloading studies from subset 2")
    subprocess.run(["nsrr", "download", "shhs/polysomnography/edfs/shhs2", "--fast"])
    logger.info(f"Changing directory back to {current_dir}")
    os.chdir(current_dir)
    logger.info("Download complete!")

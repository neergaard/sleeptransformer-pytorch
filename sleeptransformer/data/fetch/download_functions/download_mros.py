import os
import subprocess

from sleeptransformer.utils.logger import get_logger

logger = get_logger()


def download_mros(out_dataset_folder: str, *args, **kwargs):
    """ Download the MrOS dataset from NSRR.
    This function will download the standardized NSRR .xml files and the EDFs from both visits,
    and also the dataset files containing metadata, demographics, etc.
    """
    logger.info("==================================")
    logger.info("DOWNLOADING MrOS DATASET")
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
    subprocess.run(["nsrr", "download", "mros/datasets", "--fast"])
    logger.info("Downloading annotations from visit 1")
    subprocess.run(["nsrr", "download", "mros/polysomnography/annotations-events-nsrr/visit1", "--fast"])
    logger.info("Downloading studies from visit 1")
    subprocess.run(["nsrr", "download", "mros/polysomnography/edfs/visit1", "--fast"])
    logger.info("Downloading annotations from visit 2")
    subprocess.run(["nsrr", "download", "mros/polysomnography/annotations-events-nsrr/visit2", "--fast"])
    logger.info("Downloading studies from visit 2")
    subprocess.run(["nsrr", "download", "mros/polysomnography/edfs/visit2", "--fast"])
    logger.info(f"Changing directory back to {current_dir}")
    os.chdir(current_dir)
    logger.info("Download complete!")

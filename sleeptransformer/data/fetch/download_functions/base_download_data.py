# import argparse
# import os
# import subprocess

# import mne

# from nsrr_data.utils.logger import get_logger

# logger = get_logger()


# def fetch_subject(subject, path, runs=range(1, 15)):
#     """Download a single subject from the EEGBCI dataset."""
#     return mne.datasets.eegbci.load_data(subject, runs, path)


# def download_mros(out_dataset_folder, n_first=None):
#     """Download the MrOS dataset from NSRR.
#     The script will download the standardized NSRR .xml files and the EDFs from both visits,
#     and also the dataset files containing metadata, demographics, etc.
#     """

#     # Make directory
#     if not os.path.exists(out_dataset_folder):
#         logger.info(f"Creating output directory {out_dataset_folder}")
#         os.makedirs(out_dataset_folder)

#     # Use the NSRR Ruby gem to download the dataset
#     current_dir = os.getcwd()
#     logger.info(f"Changing directory to {out_dataset_folder}")
#     os.chdir(out_dataset_folder)
#     logger.info("Downloading metadata and demographics")
#     subprocess.run(["nsrr", "download", "mros/datasets", "--shallow"])
#     logger.info("Downloading annotations from visit 1")
#     subprocess.run(["nsrr", "download", "mros/polysomnography/annotations-events-nsrr/visit1", "--shallow"])
#     logger.info("Downloading annotations from visit 2")
#     subprocess.run(["nsrr", "download", "mros/polysomnography/annotations-events-nsrr/visit2", "--shallow"])
#     logger.info("Downloading studies from visit 1")
#     subprocess.run(["nsrr", "download", "mros/polysomnography/edfs/visit1", "--shallow"])
#     logger.info("Downloading studies from visit 2")
#     subprocess.run(["nsrr", "download", "mros/polysomnography/edfs/visit2", "--shallow"])
#     logger.info(f"Changing directory back to {current_dir}")
#     os.chdir(current_dir)
#     logger.info("Download complete!")

#     # # Get subjects
#     # for n in range(n_first):
#     #     fetch_subject(n + 1, out_dataset_folder)


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-o",
#         "--output_dir",
#         type=str,
#         default="./data",
#         help="Path to output directory.\nWill be created if not available.",
#     )
#     parser.add_argument("-n", "--n_first", type=int, help="Number of recordings to download.")
#     args = parser.parse_args()
#     download_mros(args.output_dir, args.n_first)

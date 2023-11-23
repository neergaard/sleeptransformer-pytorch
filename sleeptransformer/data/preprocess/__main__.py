import argparse
import sys
from pathlib import Path

from sleeptransformer.data.preprocess.process_functions import process_fns
from sleeptransformer.utils.logger import get_logger

AVAILABLE_COHORTS = set(process_fns.keys())
logger = get_logger()


def process_cohort(cohort, *args, **kwargs):
    process_fns[cohort](*args, **kwargs)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cohort', type=str, required=True, choices=AVAILABLE_COHORTS, help='Available cohorts.')
    parser.add_argument("-d", "--data_dir", type=Path, required=True, help="Path to EDF data.")
    parser.add_argument("-o", "--output_dir", type=Path, default=True, help="Where to store H5 files.")
    parser.add_argument("--fs", type=int, default=128, help="Desired resampling frequency.")
    parser.add_argument("--subjects", type=int, default=None, help='Number of subjects to process. If None, all are processed.')
    parser.add_argument('--splits', type=int, default=1, help="If processing on multiple computers, use this parameter to control the total number of splits.")
    parser.add_argument('--current_split', type=int, default=1, help="Use this to indicate the current split out of the total number of splits.")
    parser.add_argument('--duration', type=float, default=None, help='Duration of segments in seconds.')
    parser.add_argument('--overlap', type=float, default=None, help='Duration of overlap between segments in seconds.')
    parser.add_argument('--transform', type=str, default=None, choices=['none', 'stft', 'cwt', 'multitaper'], help='Type of time-frequency transformation to perform.')
    args = parser.parse_args()
    # fmt: on

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------------------------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == (len(vars(args)) - 1):
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    process_cohort(**vars(args))

import argparse
import sys
from typing import Optional

from sleeptransformer.data.fetch import download_fns
from sleeptransformer.utils.logger import get_logger

logger = get_logger()

AVAILABLE_COHORTS = set(download_fns.keys())


def download_cohort(
    output_dir: str,
    cohort: str,
):
    download_fns[cohort](output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m nsrr_data.fetch_data", description="Download NSRR data with access token."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="./data",
        help="Path to output directory.\nWill be created if not available.",
    )
    parser.add_argument(
        "-c", "--cohort", type=str, choices=AVAILABLE_COHORTS, help="Available cohorts.", required=True
    )
    args = parser.parse_args()

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------------------------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == (len(vars(args)) - 1):
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    download_cohort(args.output_dir, args.cohort)

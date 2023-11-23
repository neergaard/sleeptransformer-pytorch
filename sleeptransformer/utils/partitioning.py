import os
import random
from pathlib import Path
from typing import Union, Optional

from sleeptransformer.utils.logger import get_logger

logger = get_logger()


def get_train_validation_test(
    h5_directory: Union[list, Path],
    percent_test: Optional[float] = None,
    percent_validation: Optional[float] = None,
    number_test: Optional[int] = None,
    number_validation: Optional[int] = None,
    n_records: Optional[int] = None,
    seed=None,
) -> dict:
    if isinstance(h5_directory, list):
        records = h5_directory
    elif isinstance(h5_directory, Path):
        records = list(h5_directory.rglob("*.h5"))
    else:
        records = [x for x in os.listdir(h5_directory) if x != ".cache"]
    records = sorted(records)[:n_records]
    assert len(records) > 0, "No records found!"
    logger.info(f'Found {len(records)} records, shuffling...')
    random.seed(seed)
    random.shuffle(records)
    if number_test is not None:
        index_test = number_test
    elif percent_test is not None:
        index_test = int(len(records) * percent_test / 100)
    else:
        ValueError("Please supply either the number or percentage of test examples!")
    logger.info(f'Using {index_test} records for testing.')
    test = records[:index_test]
    records_train = records[index_test:]
    random.shuffle(records_train)
    if number_validation is not None:
        index_validation = number_validation
    elif percent_validation is not None:
        index_validation = int(len(records_train) * percent_validation / 100)
    else:
        ValueError("Please supply either the number or the percentage of validation examples!")
    logger.info(f'Using {index_validation} records for validation.')
    validation = records_train[:index_validation]
    logger.info(f'Using {len(records_train) - index_validation} records for training.')
    train = records_train[index_validation:]
    return {"train": train, "eval": validation, "test": test}

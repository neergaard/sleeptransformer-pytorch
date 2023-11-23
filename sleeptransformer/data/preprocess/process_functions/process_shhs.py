import json
import os
from pathlib import Path
from typing import Literal, Optional

import h5py
import mne
import numpy as np
import pandas as pd
import xmltodict
from joblib import delayed, parallel_backend
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

from sleeptransformer.data import transforms
from sleeptransformer.utils.filters import ButterworthFilter
from sleeptransformer.utils.logger import get_logger
from sleeptransformer.utils.parallel_bar import ParallelExecutor

logger = get_logger()

PSG_DIRECTORY = Path("polysomnography")
EDF_DIRECTORY = PSG_DIRECTORY / "edfs"
EDF_VISIT1_DIRECTORY = EDF_DIRECTORY / "shhs1"
EDF_VISIT2_DIRECTORY = EDF_DIRECTORY / "shhs2"
ANNOTATION_DIRECTORY = PSG_DIRECTORY / "annotations-events-nsrr"
ANNOTATION_VISIT1_DIRECTORY = ANNOTATION_DIRECTORY / "shhs1"
ANNOTATION_VISIT2_DIRECTORY = ANNOTATION_DIRECTORY / "shhs2"
APPLIED_CHANNELS = ["C3", "C4", "EOGL", "EOGR", "EMG"]
CHANNEL_ORDER = ["C3", "C4", "EOGL", "EOGR", "EMG"]
EXTRACTED_CHANNELS = ["C3", "C4", "EOGL", "EOGR", "EMG"]
eeg_filter = ButterworthFilter(order=2, fc=[0.3, 35], type="band")
eog_filter = ButterworthFilter(order=2, fc=[0.3, 35], type="band")
emg_filter = ButterworthFilter(order=4, fc=[10], type="highpass")
signal_labels_json_path = Path("data/hpc/montage_code/shhs.json")
if signal_labels_json_path.exists():
    with open(signal_labels_json_path, "r") as (f):
        channel_dict = json.load(f)
    # channel_categories = channel_dict["categories"]
    channel_categories = ["C4"]
channel_filters = {
    "C3": eeg_filter,
    "C4": eeg_filter,
    "EOGL": eog_filter,
    "EOGR": eog_filter,
    "EMG": emg_filter,
}


def get_events_start_duration(event_data: pd.DataFrame, event: tuple):
    start = []
    duration = []
    for ev in event[1].keys():
        for s in event[1][ev]["string"]:
            if event_data.query(f'{ev} == "{s}"').shape[0] != 0:
                start.append(event_data.query(f'{ev} == "{s}"').Start.values.astype(np.float))
                duration.append(event_data.query(f'{ev} == "{s}"').Duration.values.astype(np.float))

        if start:
            if duration:
                start = np.concatenate(start)
                duration = np.concatenate(duration)

    return {"label_idx": event[1][ev]["label"], "start": start, "duration": duration}


def process_file(
    record: str,
    record_directory: str,
    annotation_directory: str,
    output_dir: str,
    output_fs: int,
    transform: Literal["none", "stft", "cwt", "multitaper"] = "none",
    duration: Optional[float] = None,
    overlap: Optional[float] = None,
):
    assert (duration is None and overlap is None) or (
        duration is not None and overlap is not None
    ), f"'duration' and 'overlap' params must both be specified or None, received 'duration'={duration} and 'overlap'={overlap}."

    edf_filename = record_directory / Path(record).with_suffix(".edf")
    annotation_filename = annotation_directory / Path(record + "-nsrr").with_suffix(".xml")
    h5_filename = output_dir / Path(record).with_suffix(".h5")
    # if h5_filename.exists() and h5_filename.stat().st_size > 1e6:
    #     return 0
    try:
        with open(annotation_filename, "r", encoding="utf-8") as f:
            xml = f.read()
    except FileNotFoundError:
        logger.info(f"File not found, skipping file: {annotation_filename}")
        return -1
    annotations = xmltodict.parse(xml)
    annotations = pd.DataFrame(annotations["PSGAnnotation"]["ScoredEvents"]["ScoredEvent"])

    header = mne.io.read_raw_edf(edf_filename, verbose=False)
    labels = header.ch_names
    data = {
        k: mne.io.read_raw_edf(
            edf_filename,
            verbose=False,
            include=[ch for ch in labels if ch in channel_dict[k]],
        )
        for k in channel_categories
    }

    try:
        fs = {k: data[k].info["sfreq"] for k in data.keys()}
        data = {k: data[k].get_data() for k in data.keys()}
    except ValueError as err:
        print(edf_filename)
        print(str(err))
        return 0

    # Resample and filter
    for chn in data.keys():
        data[chn] = resample_poly(data[chn], output_fs, fs[chn], axis=1)
        data[chn] = channel_filters[chn](data[chn], output_fs)

    # Write to H5
    with h5py.File(h5_filename, "w") as h5:
        # Get sleep stage annotations
        stages_df = annotations.query('EventType == "Stages|Stages"')[["EventConcept", "Start", "Duration"]]
        stages = [
            (stage, start, dur)
            for stage, start, dur in zip(
                stages_df["EventConcept"].str.split("|", expand=True)[0].str.split(" sleep", expand=True)[0].to_list(),
                stages_df["Start"].astype(float).to_list(),
                stages_df["Duration"].astype(float).to_list(),
            )
        ]

        # Save sleep stages in dense format.
        # Here, we map {W, N1, N2, N3, R} to {0, 1, 2, 3, 4}
        stage_dict = {
            "Wake": 0,
            "Stage 1": 1,
            "Stage 2": 2,
            "Stage 3": 3,
            "Stage 4": 3,
            "REM": 4,
            "MOVEMENT": 7,
            "Movement": 7,
            "UNKNOWN": 7,
            "Unscored": 7,
        }
        stages_dense = np.concatenate([np.repeat(stage_dict[s[0]], s[-1]) for s in stages])
        # h5.create_dataset(f"stages", data=stages_dense)

        info = mne.create_info(list(data.keys()), output_fs, verbose=False)
        d = np.concatenate([v for v in data.values()])
        stage_annot = mne.Annotations(
            onset=[s[1] for s in stages],
            duration=[s[2] for s in stages],
            description=[s[0] for s in stages],
        )

        annotation_desc_2_event_id = {
            "Wake": 0,
            "Stage 1": 1,
            "Stage 2": 2,
            "Stage 3": 3,
            "REM": 4,
        }

        # Crop all wake except 30 min before and after
        if stage_annot[0]["description"] == "Wake":
            crop_prior = True
        else:
            crop_prior = False
        if stage_annot[-1]["description"] == "Wake":
            crop_post = True
        else:
            crop_post = False
        stage_annot.crop(
            max(0, stage_annot[1]["onset"] - 30 * 60) if crop_prior else None,
            min(stage_annot.duration.sum(), stage_annot[-2]["onset"] + 30 * 60) if crop_post else None,
        )

        # Create Raw object and set annotations
        raw = mne.io.RawArray(d, info, verbose=False)
        raw.set_annotations(stage_annot)

        # Define events
        stage_events, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30.0)
        event_id = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

        # Create Epochs object based on events
        epoched = mne.Epochs(
            raw=raw,
            events=stage_events,
            event_id={k: v for k, v in event_id.items() if v in np.unique(stage_events[:, -1])},
            tmin=0,
            tmax=30.0 - 1 / raw.info["sfreq"],
            baseline=None,
        )

        # Possibly transform data
        X = epoched.get_data()
        N, C, T = X.shape
        assert C == 1, f"This currently only works for single EEG, got C={C}."
        if transform == "stft":
            transformer = transforms.STFTTransform(
                fs=output_fs, segment_size=2 * output_fs, step_size=output_fs, nfft=256
            )
            X = transformer(X)

        # Save everything to disk
        dshape = X.shape
        h5.create_dataset(
            "data/unscaled",
            data=X,
            chunks=(1,) + dshape[1:],  # Chunk along the N dimension
        )
        h5.create_dataset("stages", data=epoched.events[:, -1], chunks=(1,))
        h5.create_group("data/channel_idx")
        h5.create_group("data/fs")
        h5.create_group("data/fs_orig")
        for idx, chn in enumerate(data.keys()):
            h5["data/channel_idx"].attrs[chn] = idx
            h5["data/fs"].attrs[chn] = output_fs
            h5["data/fs_orig"].attrs[chn] = fs[chn]
    return 0


def process_shhs(
    data_dir: Path,
    output_dir: Path,
    fs: int,
    subjects: Optional[int],
    splits: int,
    current_split: int,
    *args,
    **kwargs,
):
    logger.info("Converting EDF and annotations to standard H5 file")
    logger.info(f"Input directory (EDF and annotation file location): {data_dir}")
    logger.info(f"Output directory (H5 file location): {output_dir}")

    if not output_dir.exists():
        logger.info(f"Creating directory: {output_dir}")
        output_dir.mkdir(exist_ok=True, parents=True)

    record_directory = data_dir / EDF_VISIT1_DIRECTORY
    annotation_directory = data_dir / ANNOTATION_VISIT1_DIRECTORY
    records = sorted([x.split(".")[0] for x in os.listdir(record_directory) if x[-3:] == "edf"])[:subjects]

    records_splits = [list(s) for s in np.array_split(records, splits)]

    with parallel_backend("loky", inner_max_num_threads=2):
        ParallelExecutor(n_jobs=1)(total=(len(records_splits[current_split - 1])))(
            delayed(process_file)(
                record,
                record_directory,
                annotation_directory,
                output_dir,
                fs,
                *args,
                **kwargs,
            )
            for record in records_splits[current_split - 1]
        )

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

from sleeptransformer.utils.filters import ButterworthFilter
from sleeptransformer.utils.logger import get_logger
from sleeptransformer.utils.parallel_bar import ParallelExecutor

logger = get_logger()

PSG_DIRECTORY = "polysomnography"
EDF_DIRECTORY = os.path.join(PSG_DIRECTORY, "edfs")
EDF_VISIT1_DIRECTORY = os.path.join(EDF_DIRECTORY, "visit1")
EDF_VISIT2_DIRECTORY = os.path.join(EDF_DIRECTORY, "visit2")
ANNOTATION_DIRECTORY = os.path.join(PSG_DIRECTORY, "annotations-events-nsrr")
ANNOTATION_VISIT1_DIRECTORY = os.path.join(ANNOTATION_DIRECTORY, "visit1")
ANNOTATION_VISIT2_DIRECTORY = os.path.join(ANNOTATION_DIRECTORY, "visit2")
EVENT_CONCEPTS = dict(
    ar=dict(EventType=dict(string=["Arousals|Arousals"], label=1)),
    lm=dict(EventType=dict(string=["Limb Movement|Limb Movement"], label=2)),
    sdb=dict(
        EventConcept=dict(
            string=[
                "Obstructive apnea|Obstructive Apnea",
                "Hypopnea|Hypopnea",
                "Unsure|Unsure",
                "Central apnea|Central Apnea",
            ],
            label=3,
        )
    ),
)
APPLIED_CHANNELS = ["C3", "C4", "EOGL", "EOGR", "CHIN", "LEGL", "LEGR", "NASAL", "THOR", "ABDO"]
CHANNEL_ORDER = ["C3", "C4", "EOGL", "EOGR", "CHIN", "LEGL", "LEGR", "NASAL", "THOR", "ABDO"]
EXTRACTED_CHANNELS = [
    "A1",
    "A2",
    "C3",
    "C4",
    "EOGL",
    "EOGR",
    "LChin",
    "RChin",
    "LegL",
    "LegR",
    # "Airflow",
    "NasalP",
    "Thor",
    "Abdo",
    # "ECG L",
    # "ECG R",
]
eeg_filter = ButterworthFilter(order=2, fc=[0.3, 35], type="band")
eog_filter = ButterworthFilter(order=2, fc=[0.3, 35], type="band")
emg_filter = ButterworthFilter(order=4, fc=[10], type="highpass")
nasal_filter = ButterworthFilter(order=4, fc=[0.03], type="highpass")
belt_filter = ButterworthFilter(order=2, fc=[0.1, 15], type="band")
signal_labels_json_path = Path("nsrr_data/preprocessing/montage/mros.json")
# assert os.path.exists(signal_labels_json_path)
if signal_labels_json_path.exists():
    with open(signal_labels_json_path, "r") as (f):
        channel_dict = json.load(f)
    channel_categories = channel_dict["categories"]
channel_filters = {
    "C3": eeg_filter,
    "C4": eeg_filter,
    "EOGL": eog_filter,
    "EOGR": eog_filter,
    "CHIN": emg_filter,
    "LEGL": emg_filter,
    "LEGR": emg_filter,
    "NASAL": nasal_filter,
    "THOR": belt_filter,
    "ABDO": belt_filter,
    "AIRFLOW": belt_filter,
}
EVENT_CHANNELS = dict(ar=["C3", "C4", "EOGL", "EOGR", "CHIN"], lm=["LEGL", "LEGR"], sdb=["NASAL", "THOR", "ABDO"])


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
    duration: Optional[float] = None,
    overlap: Optional[float] = None,
    event_type: Optional[Literal["ar", "lm", "sdb"]] = None,
):
    assert (duration is None and overlap is None) or (
        duration is not None and overlap is not None
    ), f"'duration' and 'overlap' params must both be specified or None, received 'duration'={duration} and 'overlap'={overlap}."

    edf_filename = os.path.join(record_directory, record + ".edf")
    annotation_filename = os.path.join(annotation_directory, record + "-nsrr.xml")
    h5_filename = os.path.join("{}".format(output_dir), "{}.h5".format(record))
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
        k: mne.io.read_raw_edf(edf_filename, verbose=False, exclude=[ch for ch in labels if ch not in channel_dict[k]])
        for k in channel_categories
    }

    try:
        fs = {k: data[k].info["sfreq"] for k in data.keys()}
        data = {k: data[k][:][0] for k in data.keys()}
    except ValueError as err:
        print(edf_filename)
        print(str(err))
        return 0
    data["C3"] = data["C3"] - data["A2"]
    data["C4"] = data["C4"] - data["A1"]
    data["EOGL"] = data["EOGL"] - data["A2"]
    data["EOGR"] = data["EOGR"] - data["A2"]
    data["LEGL"] = data["LegL"]
    data["LEGR"] = data["LegR"]
    data["CHIN"] = data["LChin"] - data["RChin"]
    data["NASAL"] = data["NasalP"]
    data["THOR"] = data["Thor"]
    data["ABDO"] = data["Abdo"]
    fs["LEGL"] = fs["LegL"]
    fs["LEGR"] = fs["LegR"]
    fs["CHIN"] = fs["LChin"]
    fs["NASAL"] = fs["NasalP"]
    fs["THOR"] = fs["Thor"]
    fs["ABDO"] = fs["Abdo"]
    for k in ("A1", "A2", "LegL", "LegR", "LChin", "RChin", "NasalP", "Thor", "Abdo"):
        data.pop(k, None)
        fs.pop(k, None)

    # Depending on the event type, select appropriate channels
    if event_type:
        for remove_chn in data.keys() - EVENT_CHANNELS[event_type]:
            data.pop(remove_chn, None)
            fs.pop(remove_chn, None)

    # Resample and filter
    for chn in data.keys():
        data[chn] = resample_poly(data[chn], output_fs, fs[chn], axis=1)
        data[chn] = channel_filters[chn](data[chn], output_fs)

    # Write to H5
    with h5py.File(h5_filename, "w") as h5:
        if event_type:
            event_concept = (event_type, EVENT_CONCEPTS[event_type])
            events = get_events_start_duration(annotations, event_concept)
            h5.create_dataset(f"events/{event_type}/start", data=events["start"])
            h5.create_dataset(f"events/{event_type}/duration", data=events["duration"])
            h5[f"events/{event_type}"].attrs["idx"] = events["label_idx"]
        else:
            for event_concept in EVENT_CONCEPTS.items():
                events = get_events_start_duration(annotations, event_concept)
                # h5.create_group(event_concept[0])
                h5.create_dataset(f"events/{event_concept[0]}/start", data=events["start"])
                h5.create_dataset(f"events/{event_concept[0]}/duration", data=events["duration"])
                h5[f"events/{event_concept[0]}"].attrs["idx"] = events["label_idx"]

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
            "UNKNOWN": 7,
            "Unscored": 7,
        }
        stages_dense = np.concatenate([np.repeat(stage_dict[s[0]], s[-1]) for s in stages])
        h5.create_dataset(f"stages", data=stages_dense)

        if (duration is None) and (overlap is None):
            for chn in data.keys():
                x_scaled = RobustScaler().fit_transform(data[chn].T).T

                # h5.create_dataset(f"data/unscaled/{chn.lower()}", data=data[chn].squeeze())
                h5.create_dataset(f"data/scaled/{chn.lower()}", data=x_scaled.squeeze())
                h5.create_dataset(f"data/fs/original/{chn.lower()}", data=fs[chn])
                h5.create_dataset(f"data/fs/new/{chn.lower()}", data=output_fs)
        else:
            info = mne.create_info(list(data.keys()), output_fs, verbose=False)
            d = np.concatenate([v for v in data.values()])
            d_scaled = RobustScaler().fit_transform(d.T).T
            raw = mne.io.RawArray(d, info, verbose=False)
            raw_scaled = mne.io.RawArray(d_scaled, info, verbose=False)
            epoched = mne.make_fixed_length_epochs(raw, duration=duration, overlap=overlap, proj=False, verbose=False)
            epoched_scaled = mne.make_fixed_length_epochs(
                raw_scaled, duration=duration, overlap=overlap, proj=False, verbose=False
            )
            # h5.create_dataset(
            #     "data/unscaled",
            #     data=epoched.get_data(),
            #     chunks=(1, len(data.keys()), duration * output_fs),
            # )
            h5.create_dataset(
                "data/scaled",
                data=epoched_scaled.get_data(),
                chunks=(1, len(data.keys()), duration * output_fs),
                dtype="f4",
            )
            # h5.create_dataset(f"data/fs/{chn.lower()}", data=fs[chn])
            h5.create_group("data/channel_idx")
            h5.create_group("data/fs")
            h5.create_group("data/fs_orig")
            for idx, chn in enumerate(data.keys()):
                h5["data/channel_idx"].attrs[chn] = idx
                h5["data/fs"].attrs[chn] = output_fs
                h5["data/fs_orig"].attrs[chn] = fs[chn]
            # for chn in data.keys():
            #     h5.create_dataset(
            #         f"data/unscaled/{chn.lower()}",
            #         data=epoched.get_data(chn).squeeze(),
            #         chunks=(1, duration * output_fs),
            #     )
            #     h5.create_dataset(
            #         f"data/scaled/{chn.lower()}",
            #         data=epoched_scaled.get_data(chn).squeeze(),
            #         chunks=(1, duration * output_fs),
            #     )
            #     h5.create_dataset(f"data/fs/{chn.lower()}", data=fs[chn])
            #     h5["data/fs"].attrs["fs"] = output_fs
    # logger.info(f"Subject {h5_filename} written to disk")
    return 0


def process_mros(
    data_dir: str, output_dir: str, fs: int, subjects: Optional[int], splits: int, current_split: int, *args, **kwargs
):
    logger.info("Converting EDF and annotations to standard H5 file")
    logger.info(f"Input directory (EDF and annotation file location): {data_dir}")
    logger.info(f"Output directory (H5 file location): {output_dir}")

    if not os.path.exists(output_dir):
        logger.info(f"Creating directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    record_directory = os.path.join(data_dir, EDF_VISIT1_DIRECTORY)
    annotation_directory = os.path.join(data_dir, ANNOTATION_VISIT1_DIRECTORY)
    records = sorted([x.split(".")[0] for x in os.listdir(record_directory) if x[-3:] == "edf"])[:subjects]

    records_splits = [list(s) for s in np.array_split(records, splits)]

    with parallel_backend("loky", inner_max_num_threads=2):
        ParallelExecutor(n_jobs=4)(total=(len(records_splits[current_split - 1])))(
            delayed(process_file)(record, record_directory, annotation_directory, output_dir, fs, *args, **kwargs)
            for record in records_splits[current_split - 1]
        )

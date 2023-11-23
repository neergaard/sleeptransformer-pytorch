from typing import Union

import numpy as np
from h5py import File

EVENT_NAMES = {"ar": "Arousal", "lm": "Leg Movement", "sdb": "Sleep-disordered Breathing"}
EVENT_CODES = {"ar": 0, "lm": 1, "sdb": 2}


def get_record_metadata(filename: str, events: dict, fs: int, window_size: int = None):
    event_data = {k: {"data": None, "label": None, "name": None} for k in events.keys()}
    # Get signal metadata
    with File(filename, "r") as h5:
        # This gets the event data
        events_in_file = h5["events"].keys()
        for event in events.keys():
            if event not in events_in_file:
                continue
            inits = h5["events"][event]["start"][:]
            durs = h5["events"][event]["duration"][:]
            if event == "ar":
                durs = np.clip(durs, 3.0, None)
            elif event == "lm":
                durs = np.clip(durs, 0.5, 10.0)
            data = np.stack([inits, durs], axis=1)
            event_data[event]["data"] = np.round(data * fs).astype(int)
            event_data[event]["name"] = EVENT_NAMES[event]
            event_data[event]["label"] = EVENT_CODES[event]
        n_events = {k: ev["data"].shape[0] for k, ev in event_data.items()}

        # Get the waveforms and shape info
        # waveforms = h5["data"]["scaled"]
        # waveforms = np.stack([w for w in waveforms.values()])
        N, C, T = h5["data"]["unscaled"].shape
        # N = T // window_size if window_size is not None else 1
        stages = h5["stages"][:]

        # Get indices of windows containing events
        valid_idx = []
        # for x in range(0, N):
        start_stop = np.array([[x * window_size // 2, x * window_size // 2 + window_size] for x in np.arange(N)])
        for ev_cls in events.keys():
            for ev in event_data[event]["data"]:
                midpoint = ev[0] + ev[1] // 2
                valid_idx.extend(
                    [int(x) for x in np.argwhere((midpoint >= start_stop[:, 0]) & (midpoint <= start_stop[:, 1]))]
                )
        valid_idx = set(valid_idx)

        # Set metadata
        index_to_record = [
            {"record": filename.stem, "idx": x, "window_start": x * window_size // 2}
            for x in range(N)
            if x in valid_idx
        ]
        # index_to_record = [{"record": filename.stem, "idx": x * window_size} for x in range(N)]
        index_to_record_event = []
        for event_name, event_count in n_events.items():
            index_to_record_event.extend(
                [
                    # {"record": filename.stem, "max_index": T - window_size, "event": event_name, "idx": ev_idx}
                    {"record": filename.stem, "max_index": None, "event": event_name, "idx": ev_idx}
                    for ev_idx in range(event_count)
                ]
            )
        metadata = {"n_channels": C, "length": T, "filename": filename}

    return dict(
        event_data=(filename.stem, event_data),
        index_to_record=(filename.stem, index_to_record),
        index_to_record_event=(filename.stem, index_to_record_event),
        n_events=(filename.stem, n_events),
        metadata=(filename.stem, metadata),
        stages=(filename.stem, stages),
    )


def load_waveforms(filename: str, picks: list = None, scaled: bool = True, window: Union[int, slice] = None):
    if picks is None:
        picks = ["c3", "c4", "eogl", "eogr", "chin", "legl", "legr", "nasal", "abdo", "thor"]

    with File(filename, "r") as h5:
        if scaled:
            waveforms = h5["data"]["scaled"]
        else:
            waveforms = h5["data"]["unscaled"]
        if window:
            waveforms = waveforms[window]
            channel_idx = {k.lower(): v for k, v in h5["data"]["channel_idx"].attrs.items()}
            waveform = np.stack([waveforms[:, channel_idx[chn.lower()]] for chn in picks], axis=1)
        else:
            waveform = np.stack([waveforms[w][window] for w in picks])

    return waveform

import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import mne
from joblib import delayed

from sleeptransformer.utils.parallel_bar import ParallelExecutor


def get_edf_list(data_dir):
    p = Path(data_dir)
    if data_dir.is_dir():
        print(f"Checking {data_dir} for edf files...")
        edf_files = glob.glob(os.path.join(data_dir, "**", "*.[EeRr][DdEe][FfCc]"), recursive=True)
        print("Removing any MSLT studies...")
        edf_files = [edf for edf in edf_files if "mslt" not in os.path.basename(edf.lower())]
    else:
        print(f"{data_dir} is not a valid directory.")
        edf_files = []
    return edf_files


def get_signal_headers(edf_filename):
    header = mne.io.read_raw_edf(edf_filename, verbose=False).info
    return header.ch_names


def get_channel_labels(edf_filename):
    channel_headers = get_signal_headers(edf_filename)
    try:
        return [fields["label"] for fields in channel_headers]
    except:
        return channel_headers


def show_set_selection(label_set):
    n_cols = 4
    current_item = 0
    width = 30
    row_string = ""
    for label, count in sorted(label_set.items()):
        row_string += (f"{current_item}.".ljust(4) + f"{count}".rjust(4).ljust(5) + f"{label}").ljust(width)
        current_item = current_item + 1
        if current_item % n_cols == 0:
            print(row_string)
            row_string = ""
    if len(row_string) > 0:
        print(row_string)


# def getAllChannelLabels(data_dir):
#     edfFiles = getEDFFilenames(data_dir)
#     num_edfs = len(edfFiles)
#     if num_edfs == 0:
#         label_list = []
#     else:
#         label_set = getLabelSet(edfFiles)
#         label_list = sorted(label_set)
#     return label_set, num_edfs


def get_all_channel_labels_with_counts(edf_list):
    n_edfs = len(edf_list)
    if n_edfs == 0:
        label_list = []
    else:
        output = ParallelExecutor(n_jobs=-1, prefer="threads")(total=len(edf_list))(
            delayed(get_channel_labels)(edf) for edf in edf_list
        )
        label_set_counts = Counter([l2 for l1 in output for l2 in l1])
    return label_set_counts, n_edfs


# def getLabelSet(edfFiles):
#     label_set = set()
#     for edfFile in edfFiles:
#         # only add unique channel labels to our set`
#         label_set = label_set.union(set(get_channel_labels(edfFile)))
#     return label_set


def channel_mapper(edf_list, channel_labels, json_filename=None):

    n_edfs = len(edf_list)
    if n_edfs == 0:
        print("No file(s) found!")
    else:
        label_set_counts, _ = get_all_channel_labels_with_counts(edf_list)
        label_list = sorted(list(label_set_counts.keys()))
        print()

        if len(channel_labels) > 0:
            print(
                "Enter acceptable channel indices to use for the given identifier. \n"
                "Use spaces to separate multiple indices. \n"
                f"Total number of EDFs in directory: {n_edfs}"
            )
            print()

        show_set_selection(label_set_counts)
        print()

        if len(channel_labels) > 0:

            channel_map = {}  # dict()
            # channel_map["pathname"] = data_dir  # a string
            channel_map["edf_list"] = [str(p) for p in edf_list]  # a list
            channel_map["categories"] = channel_labels  # a list of strings

            for ch in channel_labels:
                indices = [int(num) for num in input(ch + ": ").split()]
                selected_labels = [label_list[i] for i in indices]
                print(f"Selected: {selected_labels}")
                channel_map[ch] = selected_labels

            if json_filename is not None:
                with open(json_filename, "w") as json_file:
                    json.dump(channel_map, json_file, indent=4, sort_keys=True)
                print(json.dumps(channel_map))
                print()
                print(f"JSON data written to file: {json_filename}")

            return channel_map


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", required=True, type=str, help="Location of EDF(s) to check")
    parser.add_argument(
        "-o",
        "--json_filename",
        type=str,
        help="Location of the output JSON file containing channel mappings",
    )
    parser.add_argument("-c", "--channels", required=True, nargs="+", help="List of channels to map")
    args = parser.parse_args()

    edf_list = get_edf_list(args.data_dir)

    channel_mapper(edf_list, args.channels, args.json_filename)

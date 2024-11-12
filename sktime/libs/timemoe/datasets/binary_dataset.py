#!/usr/bin/env python
import json
import os

import numpy as np

from .ts_dataset import TimeSeriesDataset


class BinaryDataset(TimeSeriesDataset):
    meta_file_name = "meta.json"
    bin_file_name_template = "data-{}-of-{}.bin"

    def __init__(self, data_path):
        if not self.is_valid_path(data_path):
            raise ValueError(f"Folder {data_path} is not a valid TimeMoE dataset.")

        self.data_path = data_path

        # load meta file
        meta_file_path = os.path.join(data_path, self.meta_file_name)
        self.meta_info = load_json_file(meta_file_path)

        self.num_sequences = self.meta_info["num_sequences"]
        self.dtype = self.meta_info["dtype"]
        self.seq_infos = self.meta_info["scales"]

        # process the start index for each file
        self.file_start_idxes = []
        s_idx = 0
        for fn, length in sorted(
            self.meta_info["files"].items(), key=lambda x: int(x[0].split("-")[1])
        ):
            self.file_start_idxes.append((os.path.join(data_path, fn), s_idx, length))
            s_idx += length
        self.num_tokens = s_idx

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        seq_info = self.seq_infos[seq_idx]
        read_info_list = self._get_read_infos_by_offset_length(
            seq_info["offset"], seq_info["length"]
        )
        out = []
        for fn, offset_in_file, length in read_info_list:
            out.append(self._read_sequence_in_file(fn, offset_in_file, length))

        if len(out) == 1:
            sequence = out[0]
        else:
            sequence = np.concatenate(out, axis=0)

        if "mean" in seq_info and "std" in seq_info:
            return sequence * seq_info["std"] + seq_info["mean"]
        else:
            return sequence

    def get_sequence_length_by_idx(self, seq_idx):
        return self.meta_info[seq_idx]["length"]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _get_read_infos_by_offset_length(self, offset, length):
        # just use naive search
        binary_read_info_list = []
        end_offset = offset + length
        for fn, start_idx, fn_length in self.file_start_idxes:
            end_idx = start_idx + fn_length
            if start_idx <= offset < end_idx:
                if end_offset <= end_idx:
                    binary_read_info_list.append([fn, offset - start_idx, length])
                    break
                else:
                    binary_read_info_list.append(
                        [fn, offset - start_idx, end_idx - offset]
                    )
                    length = end_offset - end_idx
                    offset = end_idx
        return binary_read_info_list

    def _read_sequence_in_file(self, fn, offset_in_file, length):
        sentence = np.empty(length, dtype=self.dtype)
        with open(fn, mode="rb", buffering=0) as file_handler:
            file_handler.seek(offset_in_file * sentence.itemsize)
            file_handler.readinto(sentence)
        return sentence

    @staticmethod
    def is_valid_path(data_path):
        if (
            os.path.exists(data_path)
            and os.path.isdir(data_path)
            and os.path.exists(os.path.join(data_path, "meta.json"))
        ):
            for sub in os.listdir(data_path):
                # TODO check if lack bin file
                if os.path.isfile(os.path.join(data_path, sub)) and sub.endswith(
                    ".bin"
                ):
                    return True
        return False


def load_json_file(fn):
    with open(fn, encoding="utf-8") as file:
        data = json.load(file)
        return data


def save_json_file(obj, fn):
    with open(fn, "w", encoding="utf-8") as file:
        json.dump(obj, file)

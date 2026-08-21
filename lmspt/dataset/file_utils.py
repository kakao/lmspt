# -*- coding: UTF-8 -*-
""" """
import os
import json
import torchaudio


def read_json(fpath):
    file_type = os.path.splitext(fpath)[-1]

    if file_type == '.jsonl':
        with open(fpath, 'r') as f:
            data_list = [json.loads(line) for line in f]
    elif file_type == '.json':
        with open(fpath, 'r') as f:
            data_list = json.load(f)
    else:
        raise NotImplementedError(
            f"Unknown meta file extension: \"{file_type}\". Currently, .json and .jsonl files are supported. "
        )

    return data_list


def read_lists(list_file):
    """
    :param list_file:
    :return:
    """
    lists = []
    with open(list_file, "r", encoding="utf8") as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def read_json_lists(list_file):
    """
    :param list_file:
    :return:
    """
    lists = read_lists(list_file)
    results = {}
    for fn in lists:
        with open(fn, "r", encoding="utf8") as fin:
            results.update(json.load(fin))
    return results


def load_wav(wav, target_sr):
    """
    :param wav:
    :param target_sr:
    :return:
    """
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert (
            sample_rate > target_sr
        ), "wav sample rate {} must be greater than {}".format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sr
        )(speech)
    return speech


def speed_change(waveform, sample_rate, speed_factor: str):
    """
    :param waveform:
    :param sample_rate:
    :param speed_factor:
    :return:
    """
    effects = [["tempo", speed_factor], ["rate", f"{sample_rate}"]]  # speed_factor
    augmented_waveform, new_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, effects
    )
    return augmented_waveform, new_sample_rate

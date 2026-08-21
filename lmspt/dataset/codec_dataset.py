import os
import torch
import torchaudio
import random
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
from lmspt.dataset.file_utils import read_json
from datasets import load_dataset, load_from_disk, concatenate_datasets


class AudioSemanticFeatDataset(torch.utils.data.Dataset):
    """File/manifest-based dataset.

    Each manifest record must carry an ``audio_path`` (absolute, or relative to
    ``audio_root``/``root``); audio is read from disk with ``torchaudio.load``.
    """

    def __init__(
        self,
        root=None,
        sample_rate=16000,
        output_sample_rate=None,
        segment_size=6,
        feature_extractor=None,
        manifest=None,
        audio_root=None
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate  # None = same as sample_rate
        self.segment_size_in_sec = segment_size
        self.segment_size = int(sample_rate * self.segment_size_in_sec)
        self.feature_extractor = feature_extractor
        self.audio_root = audio_root or self.root
        self.dataset = self.load_data(manifest)
        self._epoch = 0


    def set_epoch(self, epoch: int):
        # accelerate 호환
        self._epoch = epoch

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _pad_audio(audio, segment_size):
        if audio.size(-1) < segment_size:
            audio = torch.nn.functional.pad(audio, (0, segment_size - audio.size(-1)), 'constant')
        return audio

    def _get_audio(self, item):
        """Return (mono_waveform_1d, original_sample_rate) for a dataset record.

        Subclasses can override this to read audio from another source (e.g. an
        in-memory HuggingFace ``datasets`` array) while reusing all the
        segmenting / feature-extraction logic below.
        """
        audio_path = item["audio_path"]
        if not os.path.isabs(audio_path):
            assert self.audio_root is not None, (
                f"audio_path is relative ({audio_path!r}) but no root/audio_root was given"
            )
            audio_path = os.path.join(self.audio_root, audio_path)
        audio, sr = torchaudio.load(audio_path)
        return audio.mean(axis=0), sr

    def _sample_segment_teacher(self, audio, start):
        teacher_start = int(start * (self.feature_extractor.sample_rate / self.sample_rate))
        teacher_segment_size = int(self.feature_extractor.sample_rate * self.segment_size_in_sec)
        teacher_end = teacher_start + teacher_segment_size
        audio = self._pad_audio(audio[teacher_start:teacher_end], teacher_segment_size)
        return audio

    def _sample_segment(self, audio, teacher_audio=None):
        if audio.size(-1) > self.segment_size:
            max_audio_start = audio.size(-1) - self.segment_size
            start = random.randint(0, max_audio_start)
            end = start + self.segment_size
            audio = audio[start:end]
        else:
            start = 0
            end = audio.size(-1)
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(-1)), 'constant')

        if teacher_audio is not None:
            teacher_audio = self._sample_segment_teacher(teacher_audio, start)

        return audio, teacher_audio, (start, end)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        raw_mono, sr = self._get_audio(item)  # 1-D mono waveform at original sr

        model_audio = raw_mono
        if sr != self.sample_rate:
            model_audio = torchaudio.functional.resample(raw_mono, sr, self.sample_rate)

        semantic_input = None
        if self.feature_extractor:
            semantic_audio = self.feature_extractor.get_input(raw_mono, sr=sr)
            model_audio, semantic_audio, (start, end) = self._sample_segment(model_audio, teacher_audio=semantic_audio)
            semantic_input = self.feature_extractor(semantic_audio)
        else:
            model_audio, _, (start, end) = self._sample_segment(model_audio)

        result = {
            "id": idx,
            "speech": model_audio,
            "input_features": semantic_input,
            "duration": (end - start) / self.sample_rate
        }

        # Provide GT audio at output_sample_rate for loss computation.
        # Resample from ORIGINAL audio (not from the already-downsampled model_audio)
        # to preserve high-frequency information.
        if self.output_sample_rate is not None and self.output_sample_rate != self.sample_rate:
            # Compute the corresponding segment in original sample rate
            start_orig = int(start * (sr / self.sample_rate))
            gt_segment_size = int(self.segment_size_in_sec * self.output_sample_rate)
            # Resample the raw mono from original sr to output sr, then take segment
            gt_audio = raw_mono
            if sr != self.output_sample_rate:
                gt_audio = torchaudio.functional.resample(raw_mono, sr, self.output_sample_rate)
            end_orig_out = start_orig * self.output_sample_rate // sr + gt_segment_size
            start_orig_out = end_orig_out - gt_segment_size
            speech_gt = gt_audio[start_orig_out:end_orig_out]
            speech_gt = self._pad_audio(speech_gt, gt_segment_size)
            result["speech_gt"] = speech_gt

        return result

    def load_data(self, manifest):
        if type(manifest) == str:
            manifest = [manifest]

        dataset = []
        for path in manifest:
            assert os.path.exists(path), f"manifest path not found: {path}"
            if os.path.isdir(path):
                data = load_from_disk(path)
                dataset.append(data)
            else:
                data = read_json(path)
                dataset.append(Dataset.from_list(data))

        dataset = concatenate_datasets(dataset)

        return dataset


class HFAudioSemanticFeatDataset(AudioSemanticFeatDataset):
    """HuggingFace-native dataset (no local files / manifest).

    Reads audio arrays straight from a ``datasets`` dataset — download from the
    Hub and train directly. Works with any map-style audio dataset whose audio
    column decodes to ``{"array", "sampling_rate"}``, e.g.::

        HFAudioSemanticFeatDataset(
            path="openslr/librispeech_asr", name="clean", split="train.100",
            sample_rate=24000, segment_size=6, feature_extractor=...,
        )

    Notes:
    - This is map-style, so the chosen split must be fully downloadable. For very
      large / streaming-only corpora (e.g. full Emilia), load a subset or pass a
      pre-built dataset via ``hf_dataset``.
    - Extra keyword args are forwarded to ``load_dataset`` (e.g. ``cache_dir``,
      ``trust_remote_code``).
    """

    def __init__(
        self,
        path=None,
        name=None,
        split="train",
        sample_rate=16000,
        output_sample_rate=None,
        segment_size=6,
        feature_extractor=None,
        audio_column="audio",
        hf_dataset=None,
        **load_kwargs,
    ):
        self.sample_rate = sample_rate
        self.output_sample_rate = output_sample_rate
        self.segment_size_in_sec = segment_size
        self.segment_size = int(sample_rate * self.segment_size_in_sec)
        self.feature_extractor = feature_extractor
        self.audio_column = audio_column
        self._epoch = 0

        if hf_dataset is not None:
            self.dataset = hf_dataset
        else:
            assert path is not None, "Provide `path` (HF dataset id) or `hf_dataset`."
            self.dataset = load_dataset(path, name, split=split, **load_kwargs)

    def _get_audio(self, item):
        audio = item[self.audio_column]
        wav = torch.as_tensor(audio["array"], dtype=torch.float32)
        if wav.ndim > 1:  # [C, T] -> mono
            wav = wav.mean(axis=0)
        return wav, int(audio["sampling_rate"])


class AudioSemanticFeatCollator:
    def __init__(self):
        pass

    def __call__(self, data):
        ret = {
            k: [] for k in data[0].keys()
        }

        for dt in data:
            for k, v in dt.items():
                ret[k].append(v)

        for k, v_list in ret.items():
            if isinstance(v_list[0], torch.Tensor):
                ret[k] = pad_sequence(v_list, batch_first=True)

        if "duration" in ret:
            ret["duration"] = sum(ret["duration"])

        return ret

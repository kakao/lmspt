import os
import torch
import torchaudio
import whisper
import jiwer
import numpy as np
import math
from whisper_normalizer.basic import BasicTextNormalizer


def check_local_file(model_name_or_path):
    # DEFAULT_HF_HOME = "~/.cache/huggingface"
    # cache_dir = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    # file_name = os.path.join(cache_dir, "models", model_name_or_path.split("/")[-1])
    local_files_only = os.path.exists(model_name_or_path)
    file_name = model_name_or_path if local_files_only else model_name_or_path
    return local_files_only, file_name


class WhisperFeatureExtractor:
    def __init__(
        self,
        model_name,
        download_root
    ):
        whisper_model = whisper.load_model(model_name, download_root=download_root)
        self.n_mels = whisper_model.dims.n_mels
        self.hop_length = whisper.audio.HOP_LENGTH
        self.sample_rate = whisper.audio.SAMPLE_RATE
        self.max_len = 30 * self.sample_rate

    def get_input(self, audio, sr=None):
        if type(audio) == str:
            audio, sr = torchaudio.load(audio)
            audio = audio.mean(axis=0)

        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        return audio

    def time_to_mel_frames(self, start_idx, end_idx, sr):
        """Convert (start,end) time in seconds → (start,end) mel frame indices"""
        ratio = self.sample_rate / sr
        start_sample = start_idx * ratio
        end_sample = end_idx * ratio
        # pad = self.n_fft / 2
        pad = 0
        start_frame = math.floor((start_sample + pad) / self.hop_length)
        end_frame = math.ceil((end_sample + pad) / self.hop_length)
        return max(0, start_frame), end_frame

    def __call__(self, audio_or_path, sr=None):
        if type(audio_or_path) == str or sr is not None:
            audio_or_path = self.get_input(audio_or_path, sr=sr)

        T = audio_or_path.shape[-1]
        feat_chunks = []
        for _start in range(0, T, self.max_len):
            _end = min(_start + self.max_len, T)
            feat = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio_or_path[..., _start:_end]), n_mels=self.n_mels)
            feat_chunks.append(feat)

        feature = torch.cat(feat_chunks, dim=-1)
        return feature


class WhisperASRTeacher(torch.nn.Module):
    def __init__(self, model_name, sample_rate=16000, lang="en", distill_type="mse", download_root=None):
        super().__init__()
        self.model = whisper.load_model(model_name, device="cpu", download_root=download_root)
        del self.model.decoder
        self.options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=lang, task=self.options.task)
        self.sample_rate = sample_rate
        self.whisper_sample_rate = 16000
        self.n_mels = self.model.dims.n_mels
        print(f"whisper-{model_name}, n_mels: {self.n_mels}, lang: {lang}, distill_type: {distill_type}")
        print(f"whisper-sr: {self.whisper_sample_rate}, output-sr: {self.sample_rate}")

        # freeze whisper
        for p in self.model.parameters():
            p.requires_grad = False

        self.distill_type = distill_type
        self.normalizer = BasicTextNormalizer()

    def wav2mel(self, wav):
        if self.sample_rate != self.whisper_sample_rate:
            wav = torchaudio.functional.resample(wav, self.sample_rate, self.whisper_sample_rate)
        return whisper.log_mel_spectrogram(whisper.pad_or_trim(wav), n_mels=self.n_mels, device=wav.device)

    def wav2text(self, wav):
        mels = self.wav2mel(wav)
        results = whisper.decode(self.model, mels, self.options)
        texts = [r.text for r in results]
        return texts

    @torch.inference_mode()
    def calc_error_rate(self, gt_mels, predicted_wavs):
        results = whisper.decode(self.model, gt_mels, self.options)
        gt_texts = [r.text for r in results]
        pred_texts = self.wav2text(predicted_wavs)

        values = []
        for gt_text, pred_text in zip(gt_texts, pred_texts):
            gt_text = self.normalizer(gt_text)
            pred_text = self.normalizer(pred_text)
            if gt_text.strip():
                # print(f"gt_text: {gt_text}, pred_text: {pred_text}")
                values.append(min(jiwer.wer(gt_text, pred_text) * 100, 100))
            else:
                values.append(0)

        return values

    def prepare_inputs(self, mels):
        results = whisper.decode(self.model, mels, self.options)
        texts = [r.text for r in results]

        input_ids = []
        labels = []
        for text in texts:
            _input_ids = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
            input_ids.append(_input_ids)
            labels.append(_input_ids[1:] + [self.tokenizer.eot])

        label_lengths = [len(lab) for lab in labels]
        input_ids_length = [len(e) for e in input_ids]
        max_label_len = max(label_lengths + input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in
                  zip(labels, label_lengths)]
        input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in
                         zip(input_ids, input_ids_length)]  # 50257 is eot token id

        batch = {
            "labels": labels,
            "input_ids": input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False, device=mels.device) for k, v in batch.items()}

        return batch

    def forward(self, mels, predicted_wavs):
        pred_mels = self.wav2mel(predicted_wavs)
        target_features = self.model.encoder(mels).detach()
        pred_features = self.model.encoder(pred_mels)

        if self.distill_type == "l1":
            loss = torch.nn.functional.l1_loss(pred_features, target_features)
        else:
            loss = torch.nn.functional.mse_loss(pred_features, target_features)

        return loss









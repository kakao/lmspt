[![arXiv](https://img.shields.io/badge/arXiv-2506.16738-f9f107.svg?style=flat-square)](https://arxiv.org/abs/2506.16738)
[![IEEE TASLP](https://img.shields.io/badge/IEEE-TASLP%202026-00629B.svg?style=flat-square)](https://doi.org/10.1109/TASLPRO.2026.3705689)
# LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization

**LM-SPT** is a low-frame-rate neural **speech tokenizer**. It encodes speech into a **semantic** token stream — learned via **semantic speech-resynthesis distillation** — together with an **acoustic** residual-VQ stream, and reconstructs audio from the combined tokens. The resulting compact discrete tokens are designed to align well with language models for downstream speech generation.

The codebase builds on the [DualCodec](https://github.com/open-mmlab/Amphion/blob/main/models/codec/dualcodec/README.md) infrastructure.

---

## Installation

```bash
git clone https://github.com/kakao/lmspt.git
cd lmspt
pip install -e .
```

Requires Python ≥ 3.9 and a CUDA-capable GPU for training.

---

## Architecture

```
Audio ─┬─→ Semantic Encoder ─→ Semantic VQ (1 codebook)   ─┐
       |                                                   ├─→ z_q_semantic ─→ Aux Decoder ─→ semantic recon (distillation target)
       └─→ Acoustic Encoder ─→ Acoustic RVQ (N-1 codebooks)┤
                                                           ├─→ z_q ─→ Main Decoder ─→ reconstructed audio
                                              
```

- `LMSPT` (`lmspt/model_codec/lmspt_model.py`) — the codec model.
- `DualResidualVectorQuantize` (`lmspt/model_codec/quantization/rvq.py`) — splits quantization into a large semantic codebook + smaller acoustic residual codebooks.
- `SemanticResynthesisDistillationTrainer` (`lmspt/model_codec/trainer.py`) — Semantic speech-resynthesis distillation: adversarial (MPD/MSD/MRD) + mel reconstruction + Whisper distillation training.
- `WhisperASRTeacher` (`lmspt/model_codec/teacher/whisper.py`) — frozen Whisper encoder providing the semantic distillation target.

Audio sample rate is 24 kHz. Frame rate is set by the encoder stride product (e.g. `[4,5,6,8,2]` → 1920× downsample → 12.5 Hz).

---

## Configuration

Training is configured with [Hydra](https://hydra.cc/). The config tree lives under `lmspt/conf/`:

| Group | Options | Purpose |
|---|---|---|
| root | `lmspt_train.yaml` | composes everything |
| `model/` | `lmspt_12hz_8vq` | codec architecture + discriminator |
| `trainer/` | `semantic_resynthesis_trainer` | `SemanticResynthesisDistillationTrainer` |
| `teacher/` | `whisper` | distillation teacher |
| `aux_decoder/` | `aux_decoder_12hz_16k` | auxiliary semantic decoder |
| `data/` | `librispeech_hf`, `librispeech`, `emilia` | training data (`*_hf` = download from the Hub; others = local manifest) |
| `machine/` | `devbox` | **paths — edit before use** |

> ⚠️ The `machine/*.yaml` configs contain placeholder paths. **Set them to your own dataset / HF-cache locations before running.**

Override any value on the CLI, e.g. `model.model.semantic_codebook_size=16384 trainer.cfg.lambda_distill_loss=500.0`.

---

## Training

### Quick start — LibriSpeech 

`data=librispeech_hf` downloads `openslr/librispeech_asr` and trains directly:

```bash
torchrun --nproc_per_node auto --standalone train.py --config-name=lmspt_train_librispeech.yaml \
    data=librispeech_hf trainer.exp_name=my_librispeech_run
```

LibriSpeech is 16 kHz → resampled to the model's 24 kHz on the fly. 

### Reproduce — Emilia

The released model is trained on **Emilia**. Emilia is gated and multi-TB, so first materialize a local copy + per-language `meta_ds/<LANG>` index with `scripts/prepare_emilia.py`, then train with `data=emilia`:

```bash
# 0) one-time: accept terms at hf.co/datasets/amphion/Emilia-Dataset, then
huggingface-cli login
pip install datasets huggingface_hub soundfile librosa

# 1) download + build meta_ds/<LANG>  (quick starter: EN 2k utts, 50 held out for valid)
python scripts/prepare_emilia.py --out <DATA_HOME>/Emilia-Dataset \
    --languages EN --max-samples-per-lang 2000 --valid-per-lang 50
#    full multilingual: --languages EN ZH DE FR JA KO   (drop --max-samples-per-lang; many TB)

# 2) point conf/machine/*.yaml `data_home` so <DATA_HOME>/Emilia-Dataset is the --out dir;
#    in conf/data/emilia.yaml keep only the prepared languages in the train manifest, and
#    set valid_dataset.manifest to meta_ds_valid/<LANG> (or just use data=librispeech_hf for valid).

# 3) train
torchrun --nproc_per_node auto --standalone train.py --config-name=lmspt_train.yaml \
   data=emilia trainer.exp_name=my_emilia_run
```

`prepare_emilia.py` streams each language's tar shards, writes audio to `audio/<LANG>/`, and saves an `audio_path` index to `meta_ds/<LANG>` (the format `data=emilia` loads). 

For multi-node training, use your cluster's `torchrun` rendezvous settings.

---

## Inference

### Load a released checkpoint with `get_model`

Load a checkpoint by `model_id`, from the hub or a local folder:

```python
import torch, torchaudio
from lmspt.infer import get_model

# fetch weights from the hub, or pass a local folder path
model = get_model("12d5hz_v1", "hf://kakaocorp/lmspt").cuda()   # already .eval()

audio, sr = torchaudio.load("input.wav")
audio = torchaudio.functional.resample(audio, sr, 24000).reshape(1, 1, -1).cuda()

codes = model.encode(audio)          # [K, B, T] semantic + acoustic codes
recon = model.decode(codes)          # [B, 1, T] reconstructed 24 kHz audio
torchaudio.save("recon.wav", recon.squeeze(0).cpu(), 24000)
```

| `model_id` | weights file | frame rate | semantic cb | acoustic cb | VQ |
|---|---|---|---|---|---|
| `12d5hz_v1` | `lmspt_12d5hz_16384_4096.safetensors` | 12.5 Hz | 16384 | 4096 | 8 |

> The `lmspt` package must be installed to load a checkpoint — the model code isn't bundled in the published weights.

### Decoding variants

`encode` returns `codes` of shape `[K, B, T]` — `K = 8` codebooks (1 semantic + 7 acoustic). Besides full reconstruction you can decode partial / alternate paths:

```python
codes = model.encode(audio)                    # [8, B, T]

# 1) Full reconstruction — semantic + acoustic, main decoder (24 kHz)
recon = model.decode(codes)

# 2) Semantic-only reconstruction — semantic codebook only, main decoder (24 kHz)
recon_sem = model.decode(codes[:1, :, :])

# 3) Semantic reconstruction via the auxiliary decoder (16 kHz)
recon_aux = model.aux_semantic_decode(codes)   # uses the semantic codebook internally

torchaudio.save("recon.wav",     recon.squeeze(0).cpu(),     24000)
torchaudio.save("recon_sem.wav", recon_sem.squeeze(0).cpu(), 24000)
torchaudio.save("recon_aux.wav", recon_aux.squeeze(0).cpu(), 16000)  # note: 16 kHz
```

> The auxiliary decoder is the training-time semantic-distillation head and runs at **16 kHz** (`aux_decoder.sample_rate`) — save `recon_aux` at `16000`, not `24000`.

---

## License

This software is licensed under the Apache 2 license, quoted below.

Copyright 2026 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{jo2026lmspt,
  author  = {Jo, Daejin and Yun, Jeeyoung and Roh, Byungseok and Kim, Sungwoong},
  title   = {LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  year    = {2026},
  volume  = {34},
  pages   = {3714--3727},
  doi     = {10.1109/TASLPRO.2026.3705689}
}
```

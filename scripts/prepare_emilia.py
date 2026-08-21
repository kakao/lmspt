#!/usr/bin/env python
"""Download Emilia from the Hub and build the per-language ``meta_ds/<LANG>``
Arrow datasets that ``lmspt/conf/data/emilia.yaml`` expects (the
"already-downloaded" pipeline).

For each language it streams the WebDataset tar shards, writes each utterance's
audio to disk, and saves a lightweight HF-Arrow *index* (``audio_path`` +
``duration``/``sr``/``language``) via ``Dataset.save_to_disk`` — exactly the
format ``AudioSemanticFeatDataset`` loads with ``load_from_disk``.

Output layout (set ``machine.data_home`` so this dir == ``<data_home>/Emilia-Dataset``)::

    <out>/
      audio/<LANG>/<key>.wav        # materialized audio (or .mp3 with --keep-mp3)
      meta_ds/<LANG>/               # train index  (save_to_disk)
      meta_ds_valid/<LANG>/         # optional held-out valid index (--valid-per-lang)

Prereqs (Emilia is gated):
    huggingface-cli login                      # after accepting the dataset terms
    pip install datasets huggingface_hub soundfile librosa

Examples:
    # quick starter: 2000 EN utterances, hold out 50 for validation
    python scripts/prepare_emilia.py --out /data/Emilia-Dataset \
        --languages EN --max-samples-per-lang 2000 --valid-per-lang 50

    # full multilingual (huge — many TB); wav storage is ~10x mp3
    python scripts/prepare_emilia.py --out /data/Emilia-Dataset \
        --languages EN ZH DE FR JA KO
"""
import argparse
import os
from pathlib import Path

REPO = "amphion/Emilia-Dataset"


def _write_audio(sample, lang, audio_dir, keep_mp3):
    """Materialize one sample's audio to disk; return its metadata record."""
    import torch
    import torchaudio

    meta = sample.get("json", {}) or {}
    key = str(sample.get("__key__") or meta.get("id")).replace("/", "_")

    if keep_mp3:
        # requires the mp3 column cast to Audio(decode=False) -> {"bytes","path"}
        out = Path(audio_dir) / f"{key}.mp3"
        out.write_bytes(sample["mp3"]["bytes"])
        sr = int(meta.get("sr") or 24000)
    else:
        audio = sample["mp3"]  # HF Audio feature -> {"array","sampling_rate"}
        wav = torch.as_tensor(audio["array"], dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)  # [1, T]
        sr = int(audio["sampling_rate"])
        out = Path(audio_dir) / f"{key}.wav"
        torchaudio.save(str(out), wav, sr)

    return {
        "audio_path": str(out.resolve()),
        "duration": float(meta["duration"]) if meta.get("duration") is not None else None,
        "sr": sr,
        "language": lang,
    }


def build_language(lang, out_root, max_samples, valid_per_lang, source, keep_mp3, min_dnsmos):
    from datasets import load_dataset, Dataset, Audio

    audio_dir = Path(out_root) / "audio" / lang
    audio_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{source}/{lang}/*.tar"
    ds = load_dataset(REPO, data_files={lang.lower(): pattern}, split=lang.lower(), streaming=True)
    if keep_mp3:
        ds = ds.cast_column("mp3", Audio(decode=False))

    train_recs, valid_recs = [], []
    for sample in ds:
        if min_dnsmos is not None:
            dn = (sample.get("json") or {}).get("dnsmos")
            if dn is not None and float(dn) < min_dnsmos:
                continue
        rec = _write_audio(sample, lang, audio_dir, keep_mp3)
        if len(valid_recs) < valid_per_lang:
            valid_recs.append(rec)
        else:
            train_recs.append(rec)
            if max_samples and len(train_recs) >= max_samples:
                break

    meta_dir = Path(out_root) / "meta_ds" / lang
    Dataset.from_list(train_recs).save_to_disk(str(meta_dir))
    if valid_recs:
        Dataset.from_list(valid_recs).save_to_disk(str(Path(out_root) / "meta_ds_valid" / lang))
    return len(train_recs), len(valid_recs), meta_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output dir == <machine.data_home>/Emilia-Dataset")
    ap.add_argument("--languages", nargs="+", default=["EN", "ZH", "DE", "FR", "JA", "KO"],
                    help="Subset of EN ZH DE FR JA KO")
    ap.add_argument("--source", default="Emilia", choices=["Emilia", "Emilia-YODAS"])
    ap.add_argument("--max-samples-per-lang", type=int, default=None,
                    help="Cap train samples per language (omit = all — very large)")
    ap.add_argument("--valid-per-lang", type=int, default=0,
                    help="Hold out this many samples per language into meta_ds_valid/<LANG>")
    ap.add_argument("--keep-mp3", action="store_true",
                    help="Store original mp3 (needs ffmpeg at train time) instead of decoding to wav")
    ap.add_argument("--min-dnsmos", type=float, default=None, help="Drop samples below this DNSMOS")
    args = ap.parse_args()

    for lang in args.languages:
        n_tr, n_va, meta_dir = build_language(
            lang, args.out, args.max_samples_per_lang, args.valid_per_lang,
            args.source, args.keep_mp3, args.min_dnsmos,
        )
        print(f"[{lang}] train={n_tr} valid={n_va} -> {meta_dir}")

    out_abs = os.path.abspath(args.out)
    langs = " ".join(args.languages)
    print("\nDone. Next:")
    print(f"  1) Set machine.data_home so that <data_home>/Emilia-Dataset == {out_abs}")
    print(f"  2) In conf/data/emilia.yaml keep only the prepared languages [{langs}] in the train manifest.")
    if args.valid_per_lang:
        print("     Point valid_dataset.manifest at meta_ds_valid/<LANG> dirs (instead of librispeech).")
    print("  3) torchrun --nproc_per_node auto --standalone train.py  --config-name=lmspt_train.yaml \\")
    print("       model=lmspt_12hz_8vq teacher=whisper data=emilia trainer.exp_name=my_emilia_run")


if __name__ == "__main__":
    main()

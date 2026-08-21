"""Load a pretrained LM-SPT codec from a HuggingFace-hub repo or a local folder.

Example
-------
>>> from lmspt.infer import get_model
>>> model = get_model("12d5hz_v1", "hf://kakaocorp/lmspt").cuda()   # or a local dir
>>> codes = model.encode(audio)      # audio: [B, 1, T] @ 24 kHz
>>> recon = model.decode(codes)
"""
import os

# Default hub repo to pull weights from. Override via the ``pretrained_model_path``
# argument (an ``hf://<org>/<repo>`` string or a local directory).
DEFAULT_PRETRAINED_PATH = "hf://kakaocorp/lmspt"

# model_id -> weight filename inside the repo/folder
model_id_to_fname = {
    "12d5hz_v1": "lmspt_12d5hz_16384_4096.safetensors",
}
# model_id -> config file under lmspt/infer/conf/model/
model_id_to_cfgname = {
    "12d5hz_v1": "lmspt_12d5hz_16384_4096_8vq.yaml",
}

_CONF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", "model")


def _resolve_weight_path(pretrained_model_path, weight_fname):
    """Return a local path to ``weight_fname``, downloading from the hub if needed."""
    if pretrained_model_path is None:
        return None
    path = str(pretrained_model_path)
    if path.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        repo_id = path[len("hf://"):].rstrip("/")
        return hf_hub_download(repo_id=repo_id, filename=weight_fname)
    # local directory (or a direct path to the weight file)
    if os.path.isdir(path):
        return os.path.join(path, weight_fname)
    return path


def get_model(model_id="12d5hz_v1", pretrained_model_path=DEFAULT_PRETRAINED_PATH):
    """Build an ``LMSPT`` model for ``model_id`` and load its pretrained weights.

    Parameters
    ----------
    model_id : str
        Key into the registry, e.g. ``"12d5hz_v1"``.
    pretrained_model_path : str | None
        ``hf://<org>/<repo>`` to fetch from the hub, a local directory containing
        the weight file, or ``None`` to return a randomly-initialized model.
    """
    if model_id not in model_id_to_fname:
        raise KeyError(
            f"Unknown model_id {model_id!r}. Available: {list(model_id_to_fname)}"
        )

    import hydra
    from hydra import compose, initialize_config_dir

    cfgname = model_id_to_cfgname[model_id]
    with initialize_config_dir(version_base="1.3", config_dir=_CONF_DIR):
        cfg = compose(config_name=cfgname)
        model = hydra.utils.instantiate(cfg.model)

    weight_path = _resolve_weight_path(pretrained_model_path, model_id_to_fname[model_id])
    if weight_path is None:
        import warnings

        warnings.warn(
            "pretrained_model_path is None; returning a randomly-initialized model."
        )
    else:
        import safetensors.torch

        print(f"Loading LM-SPT weights from {weight_path}")
        missing, unexpected = safetensors.torch.load_model(
            model, weight_path, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"State-dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        print("LM-SPT weights loaded (strict match).")

    model.eval()
    return model

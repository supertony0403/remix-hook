"""Stem separation with demucs (htdemucs_ft by default), output at the 48 kHz working rate."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soxr

from . import config
from .audio_io import decode, read_wav, write_wav

log = logging.getLogger(__name__)


def stem_path(song: str, stem: str) -> Path:
    return config.STEMS / song / f"{stem}.wav"


def stems_exist(song: str) -> bool:
    return all(stem_path(song, s).exists() for s in config.STEM_NAMES)


def separate(song: str, src: Path, model_name: str = "htdemucs_ft", shifts: int = 2) -> None:
    """Separate `src` into drums/bass/other/vocals and store them as 48 kHz float WAVs."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.set_num_threads(max(1, (__import__("os").cpu_count() or 4)))
    model = get_model(model_name)
    model.eval()
    msr = model.samplerate
    wav = decode(src, msr).T  # (2, n)
    t = torch.from_numpy(wav)
    ref = t.mean(0)
    mean, std = ref.mean(), ref.std()
    t = (t - mean) / std
    log.info("demucs %s on %s (%.1f s, shifts=%d)", model_name, src.name, wav.shape[1] / msr, shifts)
    with torch.no_grad():
        out = apply_model(model, t[None], shifts=shifts, split=True, overlap=0.25,
                          progress=True, device="cpu")[0]
    out = out * std + mean
    for name, source in zip(model.sources, out):
        y = source.numpy().T.astype(np.float32)  # (n, 2)
        y48 = soxr.resample(y, msr, config.SR, quality="VHQ").astype(np.float32)
        write_wav(stem_path(song, name), y48, config.SR)
    # also keep the decoded original at the working rate for reference/QA
    write_wav(config.WORK / f"{song}_mix.wav", decode(src, config.SR), config.SR)


def load_stems(song: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for s in config.STEM_NAMES:
        data, sr = read_wav(stem_path(song, s))
        assert sr == config.SR, f"{song}/{s} has {sr} Hz"
        out[s] = data
    return out

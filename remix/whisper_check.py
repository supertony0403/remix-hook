"""Word-level transcription with faster-whisper from a separate venv (QA only).

faster-whisper lives in ~/Documents/Programmierung/nomissuccess-spot/.venv. Pitfall: do not
hand it a file path; it must receive float32 mono at 16 kHz. We hand over a .npy file which the
runner loads with numpy and passes as an array.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soxr

from .audio_io import to_mono

WHISPER_PY = Path.home() / "Documents/Programmierung/nomissuccess-spot/.venv/bin/python"
MODEL_DIR = "/mnt/steam-library/remix-hook/whisper-models"  # large models stay off the system disk
DEFAULT_MODEL = "turbo"  # large-v3-turbo; "small" garbles the German/English rap

_RUNNER = r"""
import json, sys
import numpy as np
from faster_whisper import WhisperModel
model = WhisperModel(sys.argv[1], device="cpu", compute_type="int8", download_root=sys.argv[3])
lang = None if sys.argv[2] == "auto" else sys.argv[2]
out = []
for path in sys.argv[4:]:
    audio = np.load(path).astype(np.float32)
    # temperature 0 only: the default fallback samples at higher temperatures -> not deterministic
    segs, _ = model.transcribe(audio, language=lang, word_timestamps=True, vad_filter=False,
                               beam_size=5, condition_on_previous_text=False, temperature=0.0)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3),
                          "p": round(w.probability, 3)})
    out.append(words)
print(json.dumps(out))
"""


def transcribe_many(clips: list[np.ndarray], sr: int, model: str = DEFAULT_MODEL,
                    language: str = "auto") -> list[list[dict]]:
    """Transcribe several clips with one model load (large models take seconds to load)."""
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, y in enumerate(clips):
            mono = to_mono(y).astype(np.float32)
            a16 = soxr.resample(mono, sr, 16_000).astype(np.float32)
            peak = float(np.abs(a16).max()) or 1.0
            npy = Path(td) / f"clip{i}.npy"
            np.save(npy, a16 / peak * 0.9)
            paths.append(str(npy))
        res = subprocess.run([str(WHISPER_PY), "-c", _RUNNER, model, language, MODEL_DIR, *paths],
                             check=True, capture_output=True, text=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def transcribe(y: np.ndarray, sr: int, model: str = DEFAULT_MODEL, language: str = "auto") -> list[dict]:
    return transcribe_many([y], sr, model, language)[0]


def norm(word: str) -> str:
    return "".join(c for c in word.lower().replace("’", "'") if c.isalnum() or c == "'")


def find_phrase(words: list[dict], phrase: str, t0: float = -1e9, t1: float = 1e9,
                fuzzy: bool = True) -> list[tuple[float, float]]:
    """All occurrences (start, end) of `phrase` whose start lies in [t0, t1].

    Fuzzy mode accepts "dont"/"don't" and tolerates one missing middle token, because Whisper
    on sung, processed vocals often drops or merges a word.
    """
    target = [norm(t) for t in phrase.split()]
    toks = [norm(w["word"]) for w in words]

    def eq(a: str, b: str) -> bool:
        return a == b or a.replace("'", "") == b.replace("'", "")

    hits: list[tuple[float, float]] = []
    n = len(target)
    for i in range(len(toks)):
        if not eq(toks[i], target[0]) and not (fuzzy and eq(toks[i], target[1] if n > 1 else "")):
            continue
        j, k, skipped = i, 0, 0
        if fuzzy and not eq(toks[i], target[0]):
            k = 1
        start_idx = i
        while k < n and j < len(toks):
            if eq(toks[j], target[k]):
                j += 1
                k += 1
            elif fuzzy and skipped == 0 and k > 0 and k < n - 1:
                k += 1
                skipped += 1
            else:
                break
        if k == n:
            s, e = words[start_idx]["start"], words[j - 1]["end"]
            if t0 <= s <= t1:
                hits.append((s, e))
    return hits

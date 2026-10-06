"""Word timestamps with faster-whisper, chunk by chunk.

faster-whisper lives in ~/Documents/Programmierung/nomissuccess-spot/.venv (same pattern as
remix/whisper_check.py, copied so the remix agent's files stay untouched). Pitfall: never hand
faster-whisper a file path; decode with an ffmpeg pipe to float32 mono 16 kHz and pass the array.

Whole stems transcribe badly (long silences, autotune: the 30 s windows drop whole refrains), so
`lyrics.timing` sends one job per arrangement clip. Run from the faster-whisper venv:

    python lyrics/whisper_run.py <jobs.json> <model> <out.json> [language]

Song B mixes German and English: with language "en" Whisper *translates* the German lines,
so `lyrics.timing` also runs "de" (keeps English words as sung) and compares.

jobs.json = [{"id": str, "wav": path, "t0": s, "t1": s, "prompt": str|null}, ...]; the output has
the same ids with word times in the wav's own time base (t0 already added back).
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np


def decode16k(path: str, t0: float, t1: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.3f}", "-t", f"{t1 - max(0.0, t0):.3f}",
         "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).copy()
    peak = float(np.abs(audio).max()) or 1.0
    return (audio / peak * 0.9).astype(np.float32)


def main() -> None:
    jobs_path, model_name, out = sys.argv[1], sys.argv[2], sys.argv[3]
    language = sys.argv[4] if len(sys.argv) > 4 else "en"
    from faster_whisper import WhisperModel

    jobs = json.load(open(jobs_path))
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=8)
    result = {"model": model_name, "language": language, "jobs": {}}
    for job in jobs:
        t0 = max(0.0, float(job["t0"]))
        audio = decode16k(job["wav"], t0, float(job["t1"]))
        segs, _ = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=False,
                                   beam_size=5, condition_on_previous_text=False,
                                   initial_prompt=job.get("prompt"))
        words = []
        for s in segs:
            for w in s.words or []:
                words.append({"word": w.word.strip(), "start": round(t0 + w.start, 3),
                              "end": round(t0 + w.end, 3), "p": round(w.probability, 3)})
        result["jobs"][job["id"]] = words
        print(job["id"], len(words), " ".join(w["word"] for w in words), flush=True)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()

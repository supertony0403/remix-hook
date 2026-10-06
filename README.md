# Remix Hook

Remix aus zwei Songs: Hook von Song A („don't you feel“) in Song B („Fall asleep…“) eingebaut, gebaut mit einer Python-Pipeline (Stems, Raster, Time-Stretch, Autotune, FX, Mix/Master) und als Spurprojekt in DaVinci Resolve Studio 21.

Ablauf, Messwerte und Entscheidungen: [docs/ARRANGEMENT.md](docs/ARRANGEMENT.md). Audio-Quellen, Stems und Renders liegen nur lokal (Rechte bei den Urhebern).

## Benutzung
```bash
uv venv --python 3.12 /mnt/steam-library/remix-hook/venv && ln -s /mnt/steam-library/remix-hook/venv .venv
uv pip install --python .venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python demucs librosa soundfile pyloudnorm pedalboard numpy scipy pytest pyworld
.venv/bin/python -m remix.bauen alles      # Stems -> Analyse -> Render/Master -> Prüfbericht
.venv/bin/python -m remix.bauen resolve    # Spuren in "Remix Hook"/"Remix" ersetzen + Audio-Render
.venv/bin/python -m pytest -q              # Unit- und Ausgabetests (-m "not slow" ohne Whisper/pYIN)
```
`work`, `stems`, `out` und `.venv` sind Symlinks nach `/mnt/steam-library/remix-hook/` (die Systemplatte ist knapp).

## Ausgabe
- `out/spuren/A1_b_instrumental.wav` … `A6_returns.wav`: durchgehende Spuren, 48 kHz, 32 bit float, gleich lang. Die Summe bei 0 dB ist der Master.
- `out/remix-hook.wav` (24 bit) und `out/remix-hook.mp3` (320 kbit/s, Titel „Don't You Feel (Remix)“).
- `out/report.json` (Mastering-Werte), `out/pruefbericht.json` (alle Prüfungen).

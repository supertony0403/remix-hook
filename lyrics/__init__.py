"""Lyric video for the remix, built as Fusion comps in DaVinci Resolve Studio 21.

Pipeline (run from the repo root with the repo venv):

1. ``python -m lyrics.timing [--whisper]`` -> ``work/lyrics/lyrics.json``: words and lines in remix
   time (faster-whisper per arrangement clip, consensus with the source stems and the original
   transcripts), kicks/snares, chops, beat and bar grid, section/comp spans.
2. ``python -m lyrics.build comps`` -> ``work/lyrics/comps/<fmt>/*.comp``: background (V1),
   one comp per section (V2), overlay with grain/flashes/light leaks (V3), for 16x9 and 9x16.
3. ``python -m lyrics.build place <fmt> background overlay Intro "Drop 1" ...``: carrier clips +
   ImportFusionComp on the timelines "Lyrics 16x9" / "Lyrics 9x16" of the project "Remix Hook".
4. ``python -m lyrics.render <fmt>``: ProRes 422 HQ master in ~/Videos/remix-hook (Media Storage),
   H.264 MP4 in out/lyrics/ plus a copy in ~/Videos, contact sheets (one frame every 2 s).
"""

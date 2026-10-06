# Remix „Don't You Feel“ × „Fall Asleep“: Arrangement

Auftrag (Anthony, 06.10.2026): „neues Projekt in DaVinci, mach krassen Remix-Track aus den beiden; nutze von dem einen die Hook und bau den anderen Song ein.“

## Quellen (nur lokal, nicht im Repo)
| | Datei | Länge | Tempo | Tonart | Rolle |
|---|---|---|---|---|---|
| A | `quelle/a_hook.mp4` | 2:33,8 | ~129,2 BPM (librosa: 64,6 = halbes Tempo) | E-Dur | **Hook-Lieferant** |
| B | `quelle/b_song.mp3` | 2:43,4 | 152 BPM | h-Moll | **Basis-Song** |

Analyse-Rohdaten: `docs/a_analyse.json`, `docs/b_analyse.json` (Beats, Segmente).

**Anpassung A → B:** Time-Stretch 129,2 → 152 BPM (Faktor ≈ 1,176) und Pitch **−2 Halbtöne** (E-Dur → D-Dur = Paralleltonart von h-Moll). Vor dem Stretch das echte A-Tempo aus dem Beat-Raster nachmessen. Gesang mit Formanterhalt (rubberband `-F` / Engine R3).

## Text-Landkarte (Whisper, Sekunden im Original)
**A (Hook):**
- 0,0–15,0 „But don't you feel, but don't you feel, but don't you feel“ ← **DER Hook**
- 46–62 Refrain: „And it's time to close what we had / No more lovin' this bad romance / And I'm fighting for my health without you here / But don't you feel ×3“
- 75–92 und 106–124 Refrain-Wiederholungen; 133–147 Outro „But don't you feel…“

**B (Basis):**
- 0–25,6 Intro (gesprochen/Tag)
- 25,6–38,2 Refrain 1: „Fall asleep waiting for my dreams to come true / Crazy man, I love you, even if it hurts so much / I want you to stay, cause I need you / All these days, yeah, where are you?“
- 38–76 Strophe 1
- 76,7–88 Refrain 2
- 88–115 Strophe 2 / Bridge
- 114,9–141 Refrain 3+4
- 141–163 Outro

## Ablauf (Takte bei 152 BPM, 1 Takt = 1,579 s), Ziel ≈ 2:40–3:00
| # | Teil | Takte | Inhalt |
|---|---|---|---|
| 1 | **Intro** | 16 | Nur B-Instrumental, Tiefpass öffnet von ~250 Hz auf voll. A-Hook „But don't you feel ×3“ trocken, nah, mit Hall-Throws am Phrasenende. Letzte 4 Takte: Riser + Snare-Roll; letzter Takt: Stutter-Chop „don't you–don't you–don't you“ (1/8 → 1/16) + **Tape-Stop**, 1 Schlag Stille. |
| 2 | **Drop 1** | 8 | B-Instrumental voll + Sub-Drop auf der Eins + A-Hook als **Vocal-Chops** („feel“ auf den Offbeats, gepitcht), Sidechain-Pumpen. |
| 3 | **B Refrain 1** | B 25,6–38,2 | Original mit B-Gesang. |
| 4 | **B Strophe 1** | gekürzt auf 16 Takte | B-Gesang + Instrumental. |
| 5 | **B Refrain 2** | B 76,7–88 | Original. Danach **Post-Chorus 4 Takte**: A „But don't you feel ×2“ über B-Drums + Bass. |
| 6 | **Breakdown / Bridge** | 16 | Drums raus, B-Bass + Flächen gefiltert. Darüber der **A-Refrain** „And it's time to close what we had / No more lovin' this bad romance / And I'm fighting for my health without you here“. Ab Takt 9 Drums zurück im Build (Kick 1/4 → 1/8 → 1/16), Riser. Letzter Takt: A „But don't you…“ und Stille vor dem Drop. |
| 7 | **Finaler Drop** | B 114,9–141 | B-Refrain 3+4 mit vollem Instrumental. In den Pausen zwischen den B-Zeilen antwortet **A „but don't you feel“** (Call-and-Response). Extra Sub, Sidechain. |
| 8 | **Outro** | 8 | Instrumental fällt in den Tiefpass, A „but don't you feel…“ mit Delay-Feedback und Hall-Fahne, sauberes Ende auf einer Eins. |

**Krass, aber musikalisch:** Jeder Übergang landet auf einer Takteins. Keine Stellen, an denen zwei Gesänge gleichzeitig über Silben stolpern, und keine Tonartkollisionen (A-Gesang immer in D-Dur). Master laut für Social/Club: **−9 bis −10 LUFS integriert, ≤ −1 dBTP**, ohne hörbares Pumpen der Stimme durch den Limiter.

## DaVinci Resolve
Neues Projekt **„Remix Hook“** (48 kHz), Timeline **„Remix“**. Jede Ebene liegt als eigene, durchgehende WAV auf einer eigenen Spur (alle bei 0 dB, Summe = Mixdown):
A1 B-Instrumental, A2 B-Gesang, A3 A-Hook-Gesang, A4 Chops/Stutter, A5 FX (Riser, Sub-Drops, Impacts), A6 Master-Hall/Delay-Returns. Marker je Abschnitt. Render: Audio-only WAV nach `~/Videos/remix-hook/` (Media Storage!), dazu MP3 320 kbit/s per ffmpeg nach `out/`.

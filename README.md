# frame-based.py — Piano Transcription + Filters + Frame-Based Chords

This script transcribes piano audio using `piano_transcription_inference`, writes MIDI/TXT/JSON, and can also generate **frame-based chord segments** (useful for “which notes were played” / near-real-time logic).

## What it does
- Loads audio (`librosa`)
- Transcribes notes + pedal (`piano_transcription_inference`)
- **Clamps** predicted note times to real audio duration (fixes impossible offsets)
- Optional post-filters: **A / B / D**
- Optional: **Frame-based chord extraction** + merging into chord segments

## Filters
### A — Adaptive consistency
Removes one-off “ghost” notes across the whole audio.
- Keeps notes that repeat (>= `--min-occurrences`) OR have enough total duration (>= `--min-total-dur-ratio-of-max`).

### B — Onset clustering (chord cleanup)
Groups notes that start almost together (a chord moment) using `--cluster-window`.
Inside each cluster it:
- Dedupe same MIDI within a short window (`--dedupe-window`)
- Dedupe octave duplicates (same pitch class like E4 + E6 → keep one “E”)
- Keep only top-K strongest notes (`--max-notes-per-cluster`)
Then globally:
- Dedupe same MIDI across nearby clusters (`--dedupe-window`)

### D — Harmonic/overtone filter
Drops likely harmonics (octaves, octave+fifth, etc.) when a lower “base” note is clearly stronger.
- Controlled by `--harmonic-velocity-ratio`

## Frame-based chord mode (real-time style)
Enable with `--write-chords`.

How it works:
1. Split time into frames of length `--frame-hop` (e.g. 0.05s = 50ms).
2. For each frame, collect “active” notes (note overlaps the frame).
3. Drop weak notes (`--frame-min-vel`) and frames with too few notes (`--frame-min-active`).
4. Merge consecutive frames into stable chord segments if they’re similar enough:
   - Similarity is Jaccard overlap ≥ `--merge-min-jaccard`
   - Drop very short segments < `--merge-min-dur`

Outputs:
- `*_chords.txt` (chord segments)
- JSON also includes `chord_segments` (and optionally `frame_chords` if enabled)

## Output files
- `*_notes.txt`  → filtered note events table
- `*_result.json` → note events + settings (+ chord segments if enabled)
- `*.mid` → MIDI output (unless `--no-midi`)
- `*_chords.txt` → frame-based chord segments (only with `--write-chords`)

## Run (your exact command)
```bash
python frame-based.py \
  --audio "/Users/alimoutabi/PycharmProjects/PythonProject6/data/test2.ogg" \
  --outdir "/Users/alimoutabi/Desktop/notes/output" \
  --enable-A --enable-B --enable-D \
  --write-chords \
  --frame-hop 0.05 \
  --frame-min-vel 20 \
  --frame-min-active 2 \
  --merge-min-jaccard 0.85 \
  --merge-min-dur 0.10
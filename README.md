# Piano Transcription (txt-format.py)

This repo contains **txt-format.py**: a small CLI tool that
- loads an audio file
- runs `piano_transcription_inference`
- clamps predicted note times to the real audio duration (important!)
- optionally applies post-filters **A / B / D**
- writes outputs: **TXT + JSON + MIDI** (unless `--no-midi`)

---

## 1) Requirements

### Python
- Python **3.10+** (3.12 works if your environment supports torch + dependencies)

### Install libraries (pip)
Recommended: use a virtualenv / conda env.

```bash
pip install --upgrade pip
pip install librosa piano-transcription-inference torch numpy soundfile audioread
```

Notes:
- If you use **.m4a** input, you may need system codecs / ffmpeg.  
  On macOS (Homebrew):
  ```bash
  brew install ffmpeg
  ```

---

## 2) What the script produces (outputs)

If your input is `test.ogg` and you set:

- `--outdir "/Users/ali/output"`

Outputs will be:
- `/Users/ali/output/test.mid`
- `/Users/ali/output/test_notes.txt`
- `/Users/ali/output/test_result.json`

---

## 3) Basic run (no filters)

Use this first to verify everything works.

```bash
python txt-format.py \
  --audio "/path/to/test.ogg" \
  --outdir "/path/to/output"
```

---

## 4) Debug run (print audio info + raw notes)

Very useful when results look wrong.

```bash
python txt-format.py \
  --audio "/path/to/test.ogg" \
  --outdir "/path/to/output" \
  --print-audio-info \
  --print-raw
```

What you should check:
- Audio duration printed matches your real file length
- RAW notes look reasonable (then filters can clean them)

---

## 5) Filters overview (A / B / D)

The model sometimes outputs **extra notes** (ghost notes, harmonics, duplicates).
Filters help reduce this.

### B = Onset clustering (core clean-up)
Groups notes by onset time (chord moment), dedupes same pitch, keeps top-K notes.

Enable with:
- `--enable-B`

Key parameters:
- `--cluster-window` (seconds)  
  Typical: `0.03` to `0.06`
- `--dedupe-window` (seconds)  
  Typical: `0.05` to `0.10`
- `--max-notes-per-cluster`  
  Typical: `3` for triads, `4-6` for richer chords

### D = Harmonic / overtone removal
Drops likely harmonics (e.g. octave/fifth above) if base note is clearly stronger.

Enable with:
- `--enable-D`

Key parameter:
- `--harmonic-velocity-ratio`  
  Typical: `1.10` to `1.30`

### A = Adaptive consistency filter (global)
Removes one-off notes across the whole clip (notes that appear only once or have tiny total duration).

Enable with:
- `--enable-A`

Key parameters:
- `--min-occurrences`  
  Typical: `2`
- `--min-total-dur-ratio-of-max`  
  Typical: `0.05` to `0.15`

---

## 6) “How to run” examples (B only / B+D / A+B+D)

### 6.1 Run with **B only**
Good first step to reduce duplicates and keep the strongest chord notes.

```bash
python txt-format.py \
  --audio "/path/to/test.ogg" \
  --outdir "/path/to/output" \
  --enable-B \
  --cluster-window 0.04 \
  --dedupe-window 0.08 \
  --max-notes-per-cluster 6
```

### 6.2 Run with **B + D**
Recommended for most “which notes were played” scenarios.

```bash
python txt-format.py \
  --audio "/path/to/test.ogg" \
  --outdir "/path/to/output" \
  --enable-B --enable-D \
  --cluster-window 0.04 \
  --dedupe-window 0.08 \
  --max-notes-per-cluster 6 \
  --harmonic-velocity-ratio 1.15
```

### 6.3 Run with **A + B + D** (strong filtering)
If you still see many ghost notes, enable A too.

```bash
python txt-format.py \
  --audio "/path/to/test.ogg" \
  --outdir "/path/to/output" \
  --enable-A --enable-B --enable-D \
  --cluster-window 0.04 \
  --dedupe-window 0.08 \
  --max-notes-per-cluster 6 \
  --harmonic-velocity-ratio 1.15 \
  --min-occurrences 2 \
  --min-total-dur-ratio-of-max 0.10
```

---

## 7) Optional flags

### Don’t keep MIDI
```bash
python txt-format.py --audio "/path/to/test.ogg" --outdir "/path/to/output" --no-midi
```

### Use CUDA (only if available)
```bash
python txt-format.py --audio "/path/to/test.ogg" --outdir "/path/to/output" --device cuda
```

---

## 8) Troubleshooting

### “Offsets longer than audio” (e.g., 6s in a 1s clip)
Fixed by the script via **clamping**:
- It clamps event offsets to the true audio duration.
- It drops notes whose onset is beyond audio duration.

### Missing C/E/G but seeing weird high notes (e.g., G6/E6)
Try:
1) Run with `--print-raw` to confirm the model outputs them
2) Enable `--enable-D` and increase harmonic filtering a bit:
   - try `--harmonic-velocity-ratio 1.25`
3) Reduce max notes per chord:
   - try `--max-notes-per-cluster 3` for a triad

### m4a loading issues
Install ffmpeg:
```bash
brew install ffmpeg
```

---

## 9) Quick “best default” command

```bash
python txt-format.py \
  --audio "/path/to/test.ogg" \
  --outdir "/path/to/output" \
  --enable-B --enable-D \
  --cluster-window 0.04 \
  --dedupe-window 0.08 \
  --max-notes-per-cluster 4 \
  --harmonic-velocity-ratio 1.15
```

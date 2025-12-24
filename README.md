# Piano Audio Transcription Tool

This tool listens to a piano audio file and detects **which notes are played over time** (including chords).
It outputs:

- a **MIDI file**
- a **TXT file** (human-readable detected notes)
- a **JSON file** (machine-readable note events)

It works with recorded audio files (`.wav`, `.ogg`, `.mp3`, `.m4a`, depending on your setup).

---

## Requirements

- Python 3.10+ (Anaconda recommended)

Install dependencies once:

```bash
pip install torch piano-transcription-inference librosa soundfile
```

---

## Basic Usage (most common)

```bash
python transcribe_piano.py --audio "/path/to/audio.ogg"
```

Outputs will be written **next to the audio file**.

---

## Specify an Output Folder

```bash
python transcribe_piano.py \
  --audio "/path/to/audio.ogg" \
  --outdir "/path/to/output"
```

---

## Command Line Arguments

| Argument | Description |
|--------|------------|
| `--audio` | **(required)** Path to the input audio file |
| `--outdir` | Output directory (default: same folder as audio) |
| `--stem` | Base name for output files (default: audio filename) |
| `--device` | Inference device: `cpu` (default) or `cuda` |
| `--no-midi` | Do not keep the generated MIDI file |
| `--full-json` | Export a JSON-safe version of the full model output (advanced) |

---

## Output Files

For an input file `A2.ogg`, the tool produces:

- `A2.mid` → MIDI transcription
- `A2_notes.txt` → readable list of detected notes
- `A2_result.json` → structured note events

### TXT example

```
idx  midi  name  onset(s)  offset(s)  dur(s)  velocity
0    60    C4    0.644     1.004      0.359   76
1    55    G3    0.645     1.010      0.365   71
```

---

## What the Tool Does (Conceptually)

1. Loads the audio file
2. Runs a piano transcription model
3. Detects:
   - which notes are played
   - when they start and end
   - multiple notes at the same time (chords)
4. Saves the results in easy-to-use formats

---

## Notes

- This tool **does not grade performance** — it only detects played notes.
- Best results are achieved with **clean piano recordings**.
- Acoustic piano via microphone is supported, but recording quality matters.

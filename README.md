# 🎹 Piano Audio Transcription (Python Only)

This tool transcribes **piano audio recordings** into:
- **MIDI** (`.mid`)
- **TXT** (human-readable detected notes)
- **JSON** (machine-readable note events)

It uses a pretrained piano transcription model and runs **locally with Python** (no Docker required).

---

## 1️⃣ Requirements

- **macOS / Linux / Windows**
- **Python 3.10 or newer**
- Internet connection (first run downloads the model ~165 MB)

Check Python version:
```bash
python3 --version
```

---

## 2️⃣ (Recommended) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ System dependency (important for audio decoding)

### macOS
Install `ffmpeg` (required for `.m4a`, `.mp3`, `.ogg`):

```bash
brew install ffmpeg
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

---

## 4️⃣ Install Python libraries

Upgrade pip and install all required libraries:

```bash
pip install --upgrade pip
pip install torch piano-transcription-inference librosa soundfile audioread numpy
```

### What these libraries do (short)
| Library | Purpose |
|------|-------|
| torch | Neural network runtime |
| piano-transcription-inference | Piano transcription model |
| librosa | Audio loading & resampling |
| soundfile | Audio decoding |
| audioread | Fallback decoder |
| numpy | Numeric arrays |

---

## 5️⃣ First run (model download)

On the **first run**, the model file (~165 MB) is downloaded automatically.
This happens only once and is reused afterwards.

---

## 6️⃣ Run the script

### Basic usage
Outputs are written **next to the audio file**.

```bash
python txt-format.py --audio "/path/to/audio.m4a"
```

---

### Specify output folder

```bash
python txt-format.py \
  --audio "/Users/alimoutabi/Desktop/notes/A.m4a" \
  --outdir "/Users/alimoutabi/Desktop/notes/output"
```

---

### Optional arguments

```bash
--device cpu        # default (use cuda only if supported)
--stem my_take      # custom output filename
--no-midi           # do not keep MIDI file
--full-json         # export full model output (advanced)
```

Example:
```bash
python txt-format.py \
  --audio "/Users/alimoutabi/Desktop/notes/A.m4a" \
  --outdir "/Users/alimoutabi/Desktop/notes/output" \
  --stem practice_01 \
  --no-midi
```

---

## 7️⃣ Output files

For input `A.m4a`, you get:

```
A.mid
A_notes.txt
A_result.json
```

TXT columns:
- idx: note index
- midi: MIDI note number
- name: note name (e.g. C4)
- onset / offset: seconds
- dur: duration
- velocity: estimated velocity

---

## 8️⃣ Troubleshooting

- **"PySoundFile failed" warning**  
  Normal — `audioread + ffmpeg` is used instead.

- **CUDA not available**  
  Use:
  ```bash
  --device cpu
  ```

- **Model download fails**  
  Check internet and write permissions in your home folder.

---

## ✔️ Summary (TL;DR)

```bash
brew install ffmpeg
python3 -m venv .venv
source .venv/bin/activate
pip install torch piano-transcription-inference librosa soundfile audioread numpy
python txt-format.py --audio "/path/to/file.m4a" --outdir "/path/to/output"
```

---

Enjoy 🎹  
This tool is designed for **practice analysis**, not grading.

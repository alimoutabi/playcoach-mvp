# 🎹 Piano Transcription (Docker)

This tool transcribes piano audio into:
- **MIDI**
- **TXT** (human-readable detected notes)
- **JSON** (structured note events)

Everything runs inside **Docker** — no local Python setup needed.

---

## 1️⃣ Requirements
- **Docker Desktop** (Mac / Windows / Linux)
  - https://www.docker.com/products/docker-desktop/

---

## 2️⃣ Project Structure
```
.
├── Dockerfile
├── docker-compose.yml
├── txt-format.py
├── requirements.txt
└── data/
    └── test.ogg
```

- Put your **audio files** into the `data/` folder
- All **outputs** will also appear in `data/`

---

## 3️⃣ Build the Docker image (first time only)
From the project root:

```bash
docker compose build
```

This will:
- install system dependencies (ffmpeg, soundfile, etc.)
- install Python packages
- download the piano transcription model (once)

---

## 4️⃣ Run transcription

### Basic usage
```bash
docker compose run --rm piano --audio /data/test.ogg
```

### Specify output folder
```bash
docker compose run --rm piano \
  --audio /data/test.ogg \
  --outdir /data/output
```

### Optional flags
```bash
--no-midi        # do not keep MIDI file
--full-json      # export full model output (advanced)
--device cpu     # default (cuda possible if GPU is supported)
```

Example:
```bash
docker compose run --rm piano \
  --audio /data/test.ogg \
  --outdir /data/output \
  --no-midi
```

---

## 5️⃣ Output files
For `test.ogg`, you will get:

```
data/
├── test.mid
├── test_notes.txt
└── test_result.json
```

---

## 6️⃣ Notes
- First run may take longer (model download ~165MB)
- Best results with clean piano recordings
- Acoustic piano via microphone is supported

---

## 7️⃣ Troubleshooting
If something goes wrong:
```bash
docker compose build --no-cache
```

---

## ✔️ That’s it
No Python install, no virtualenv, no system dependencies — just Docker.

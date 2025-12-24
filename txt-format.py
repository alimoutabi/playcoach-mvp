import json
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

#audio_path = r"/Users/alimoutabi/Desktop/notes/A.m4a"
audio_path = r"/Users/alimoutabi/Desktop/notes/A2.ogg"
out_mid = r"/Users/alimoutabi/Desktop/notes/out.mid"
out_txt = r"/Users/alimoutabi/Desktop/notes/out_notes.txt"
out_json = r"/Users/alimoutabi/Desktop/notes/out_result.json"

audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

transcriptor = PianoTranscription(device="cpu")
result = transcriptor.transcribe(audio, out_mid)

print("Result keys:", list(result.keys()))

# ✅ Your version returns notes here:
note_events = result.get("est_note_events", [])
pedal_events = result.get("est_pedal_events", [])

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"

# Sort notes by onset time
note_events = sorted(note_events, key=lambda x: float(x["onset_time"]))

lines = []
lines.append("Detected notes (est_note_events)\n")
lines.append("idx\tmidi\tname\tonset(s)\toffset(s)\tdur(s)\tvelocity\n")

for i, n in enumerate(note_events):
    onset = float(n["onset_time"])
    offset = float(n["offset_time"])
    midi = int(n["midi_note"])
    vel = n.get("velocity", "")
    dur = offset - onset
    name = midi_to_name(midi)

    lines.append(f"{i}\t{midi}\t{name}\t{onset:.3f}\t{offset:.3f}\t{dur:.3f}\t{vel}")

# --- Print to console ---
print("\n".join(lines))

# --- Also print pedal events (optional) ---
if pedal_events:
    print("\nPedal events (est_pedal_events):")
    for p in pedal_events:
        print(f"  onset={float(p['onset_time']):.3f}s  offset={float(p['offset_time']):.3f}s")

# --- Save TXT ---
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\nSaved TXT to:", out_txt)

# --- Save full result as JSON (convert numpy float32 safely) ---
def to_python(obj):
    # Convert numpy scalars to Python scalars for JSON
    try:
        import numpy as np
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
    except Exception:
        pass
    return obj

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=to_python)

print("Saved full JSON to:", out_json)
print("Wrote MIDI to:", out_mid)

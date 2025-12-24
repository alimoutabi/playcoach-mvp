#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def make_json_safe(obj, seen=None):
    """
    Convert complex objects into JSON-safe objects.
    - Converts numpy scalars/arrays
    - Converts torch tensors
    - Breaks circular references
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return "<circular_ref>"
    seen.add(obj_id)

    # numpy scalars / arrays
    try:
        import numpy as np
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # torch tensors
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # basic JSON types
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v, seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v, seen) for v in obj]

    # fallback
    return str(obj)


def build_notes_txt(note_events: list[dict]) -> str:
    note_events = sorted(note_events, key=lambda x: float(x["onset_time"]))

    lines = []
    lines.append("Detected notes (est_note_events)")
    lines.append("")
    lines.append("idx\tmidi\tname\tonset(s)\toffset(s)\tdur(s)\tvelocity")

    for i, n in enumerate(note_events):
        onset = float(n["onset_time"])
        offset = float(n["offset_time"])
        midi = int(n["midi_note"])
        vel = n.get("velocity", "")
        dur = offset - onset
        name = midi_to_name(midi)
        lines.append(f"{i}\t{midi}\t{name}\t{onset:.3f}\t{offset:.3f}\t{dur:.3f}\t{vel}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio into MIDI + note events (txt/json)."
    )
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to the input audio file (wav/mp3/ogg/m4a depending on your setup).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. If omitted, uses the audio file's folder.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (cpu or cuda).",
    )
    parser.add_argument(
        "--stem",
        default=None,
        help="Base filename for outputs (default: audio filename without extension).",
    )
    parser.add_argument(
        "--no-midi",
        action="store_true",
        help="Do not keep the MIDI file (still produces txt/json).",
    )
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="If set, writes a JSON-safe version of the full result (may include '<circular_ref>'). "
             "If not set, writes a clean minimal JSON (recommended).",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else audio_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    stem = args.stem or audio_path.stem

    out_mid = outdir / f"{stem}.mid"
    out_txt = outdir / f"{stem}_notes.txt"
    out_json = outdir / f"{stem}_result.json"

    # Load audio
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)

    # Transcribe
    transcriptor = PianoTranscription(device=args.device)

    # The library expects an out_mid path; we pass it always.
    result = transcriptor.transcribe(audio, str(out_mid))

    note_events = result.get("est_note_events", [])
    pedal_events = result.get("est_pedal_events", [])

    # Build TXT
    txt_content = build_notes_txt(note_events)

    # Console output
    print(txt_content)

    if pedal_events:
        print("Pedal events (est_pedal_events):")
        for p in pedal_events:
            print(f"  onset={float(p['onset_time']):.3f}s  offset={float(p['offset_time']):.3f}s")

    # Save TXT
    out_txt.write_text(txt_content, encoding="utf-8")

    # Save JSON (recommended: clean minimal schema)
    if args.full_json:
        payload = make_json_safe(result)
    else:
        payload = {
            "audio_file": str(audio_path),
            "note_events": note_events,
            "pedal_events": pedal_events,
        }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=make_json_safe)

    # Handle --no-midi
    if args.no_midi and out_mid.exists():
        out_mid.unlink()

    print(f"\nSaved TXT : {out_txt}")
    print(f"Saved JSON: {out_json}")
    if not args.no_midi:
        print(f"Wrote MIDI: {out_mid}")


if __name__ == "__main__":
    main()
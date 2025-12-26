#!/usr/bin/env python3
"""
txt-format.py

Piano transcription script using piano_transcription_inference.

It:
1) Loads audio with librosa
2) Runs transcription (note + pedal events)
3) Optionally post-filters notes to reduce "extra" notes
4) Saves TXT + JSON (+ MIDI unless --no-midi)

Post-filtering options:
A) Adaptive consistency filter (keeps notes that are "consistent" across the audio)
B) Onset clustering filter (group notes by onset time; keep strongest notes per cluster; dedupe)
D) Harmonic/overtone filter (drops likely harmonics like +12/+19/+24 semitones, if weaker than a base note)

IMPORTANT FIX:
- Clamp predicted note times to the real audio duration.
  This prevents offsets like 6.650s in a 1.0s audio file.

NEW (recommended for your "which notes were played" app):
- Pitch-class dedupe inside each chord cluster:
    If the model outputs E4 and E6 at the same chord moment, keep only one "E"
    (choose the one closest to the cluster's median pitch).
- Global dedupe across clusters:
    If the same MIDI is predicted multiple times within a short time window,
    keep only the strongest one.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# -----------------------------
# Helper formatting + JSON utils
# -----------------------------
def midi_to_name(midi: int) -> str:
    """Convert a MIDI note number (e.g. 60) to a note name (e.g. C4)."""
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


# -----------------------------
# Note event utilities
# -----------------------------
def note_duration(ev: dict) -> float:
    """Compute duration = offset - onset (seconds)."""
    onset = float(ev["onset_time"])
    offset = float(ev["offset_time"])
    return max(0.0, offset - onset)


def note_velocity(ev: dict) -> int:
    """
    Get velocity as int if present, otherwise 0.
    Some pipelines may omit velocity or set it None.
    """
    v = ev.get("velocity", 0)
    if v is None:
        return 0
    return int(v)


def sort_by_onset(events: list[dict]) -> list[dict]:
    """Return events sorted by onset time."""
    return sorted(events, key=lambda x: float(x["onset_time"]))


def clamp_events_to_audio(note_events: list[dict], audio_dur: float) -> list[dict]:
    """
    Ensure all note events are physically possible given the audio duration.

    What it does:
    - drops notes with onset >= audio_dur
    - clamps offset_time to audio_dur
    - ensures offset_time >= onset_time
    - returns NEW dicts (does not modify original)
    """
    clamped = []
    for ev in note_events:
        onset = float(ev["onset_time"])
        offset = float(ev["offset_time"])

        if onset >= audio_dur:
            continue  # impossible event outside audio

        offset = min(offset, audio_dur)
        offset = max(offset, onset)

        ev2 = dict(ev)
        ev2["onset_time"] = onset
        ev2["offset_time"] = offset
        clamped.append(ev2)

    return sort_by_onset(clamped)


def build_notes_txt(note_events: list[dict], title: str = "Detected notes (est_note_events)") -> str:
    """
    Create a human-readable text table of note events.
    """
    note_events = sort_by_onset(note_events)

    lines = []
    lines.append(title)
    lines.append("")
    lines.append("idx\tmidi\tname\tonset(s)\toffset(s)\tdur(s)\tvelocity")

    for i, n in enumerate(note_events):
        onset = float(n["onset_time"])
        offset = float(n["offset_time"])
        dur = note_duration(n)
        midi = int(n["midi_note"])
        vel = n.get("velocity", "")
        name = midi_to_name(midi)
        lines.append(f"{i}\t{midi}\t{name}\t{onset:.3f}\t{offset:.3f}\t{dur:.3f}\t{vel}")

    return "\n".join(lines) + "\n"


# ---------------------------------
# (B) Onset clustering (core filter)
# ---------------------------------
def cluster_by_onset(note_events: list[dict], cluster_window: float) -> list[list[dict]]:
    """
    Group note events into clusters by onset time.
    Notes are in the same cluster if their onset times are within cluster_window seconds.

    Example:
      cluster_window = 0.04  (40ms) usually groups chord notes together.

    IMPORTANT:
      We compare each onset to the CLUSTER START (not to the previous note),
      so clustering stays stable even if many notes are present.
    """
    events = sort_by_onset(note_events)
    clusters: list[list[dict]] = []

    current: list[dict] = []
    cluster_start_onset: float | None = None

    for ev in events:
        onset = float(ev["onset_time"])

        if not current:
            current = [ev]
            cluster_start_onset = onset
            continue

        if onset - float(cluster_start_onset) <= cluster_window:
            current.append(ev)
        else:
            clusters.append(current)
            current = [ev]
            cluster_start_onset = onset

    if current:
        clusters.append(current)

    return clusters


def dedupe_same_pitch_in_cluster(cluster: list[dict], dedupe_window: float) -> list[dict]:
    """
    Remove duplicates of the same MIDI pitch within a cluster.

    Why:
      Models sometimes output the same pitch multiple times very close together.
      We keep the "best" one (highest velocity, then longest duration).

    dedupe_window:
      If two notes of same midi start within dedupe_window seconds, treat as duplicates.
    """
    cluster_sorted = sort_by_onset(cluster)

    def better(a: dict, b: dict) -> dict:
        sa = (note_velocity(a), note_duration(a))
        sb = (note_velocity(b), note_duration(b))
        return a if sa >= sb else b

    by_midi: dict[int, list[dict]] = defaultdict(list)
    for ev in cluster_sorted:
        by_midi[int(ev["midi_note"])].append(ev)

    kept: list[dict] = []
    for midi, evs in by_midi.items():
        evs = sort_by_onset(evs)
        best = evs[0]

        for ev in evs[1:]:
            if abs(float(ev["onset_time"]) - float(best["onset_time"])) <= dedupe_window:
                best = better(best, ev)
            else:
                kept.append(best)
                best = ev

        kept.append(best)

    return sort_by_onset(kept)


def dedupe_pitch_class_in_cluster(cluster: list[dict]) -> list[dict]:
    """
    NEW: Remove octave-duplicates inside ONE chord cluster.

    Problem:
      The model often outputs the same note-name in different octaves at the same time
      (e.g., E4 and E6 in one chord moment). For an app that checks "which notes were played",
      we usually want only ONE representative of "E" at that moment.

    Strategy:
      - Compute the median MIDI of the cluster (represents the 'center' pitch region)
      - Group notes by pitch class (midi % 12)
      - For each pitch class, keep the note closest to the median MIDI
        (ties -> higher velocity -> longer duration)
    """
    if not cluster:
        return cluster

    mids = sorted(int(ev["midi_note"]) for ev in cluster)
    median_midi = mids[len(mids) // 2]

    by_pc: dict[int, list[dict]] = defaultdict(list)
    for ev in cluster:
        by_pc[int(ev["midi_note"]) % 12].append(ev)

    def rank(ev: dict):
        midi = int(ev["midi_note"])
        dist = abs(midi - median_midi)   # smaller is better
        vel = note_velocity(ev)          # bigger is better
        dur = note_duration(ev)          # bigger is better
        return (dist, -vel, -dur)

    kept = []
    for _, evs in by_pc.items():
        best = sorted(evs, key=rank)[0]
        kept.append(best)

    return sort_by_onset(kept)


def keep_top_k_in_cluster(cluster: list[dict], max_notes: int) -> list[dict]:
    """
    Keep only the top-K strongest notes in a cluster.

    Strength heuristic:
      Primary: velocity (descending)
      Secondary: duration (descending)

    Effective for chords: keep strongest notes, drop weak ghosts.
    """
    ranked = sorted(
        cluster,
        key=lambda ev: (note_velocity(ev), note_duration(ev)),
        reverse=True,
    )
    return sort_by_onset(ranked[:max_notes])


def dedupe_same_midi_globally(note_events: list[dict], dedupe_window: float) -> list[dict]:
    """
    NEW: Remove near-duplicate repeats of the same MIDI across the whole audio.

    Example:
      The model outputs C2 twice within 80ms -> keep only the stronger one.

    Rule:
      If same MIDI appears again within dedupe_window seconds, treat as duplicate.
      Keep the stronger one (velocity, then duration).
    """
    events = sort_by_onset(note_events)

    def better(a: dict, b: dict) -> dict:
        return a if (note_velocity(a), note_duration(a)) >= (note_velocity(b), note_duration(b)) else b

    last_by_midi: dict[int, dict] = {}
    kept: list[dict] = []

    for ev in events:
        midi = int(ev["midi_note"])
        onset = float(ev["onset_time"])

        if midi not in last_by_midi:
            last_by_midi[midi] = ev
            continue

        prev = last_by_midi[midi]
        prev_onset = float(prev["onset_time"])

        if abs(onset - prev_onset) <= dedupe_window:
            last_by_midi[midi] = better(prev, ev)
        else:
            kept.append(prev)
            last_by_midi[midi] = ev

    kept.extend(last_by_midi.values())
    return sort_by_onset(kept)


# ---------------------------------
# (D) Harmonic / overtone filtering
# ---------------------------------
def drop_likely_harmonics(
    cluster: list[dict],
    *,
    harmonic_intervals=(12, 19, 24, 31),
    min_base_velocity_ratio: float = 1.15,
) -> list[dict]:
    """
    Drop notes that look like harmonics / overtones within a chord cluster.

    Idea:
      If a high note is approximately a harmonic interval above a lower note
      (e.g. +12 semitones = octave, +19 = octave+fifth, +24 = two octaves),
      and the lower note is stronger (velocity higher by ratio),
      then the high note might be a model hallucinated overtone.

    Applied inside a single onset cluster (chord moment).
    """
    if not cluster:
        return cluster

    # If there are duplicates of same midi, keep the strongest one for harmonic checks
    best_by_midi: dict[int, dict] = {}
    for ev in cluster:
        midi = int(ev["midi_note"])
        if midi not in best_by_midi:
            best_by_midi[midi] = ev
        else:
            a = best_by_midi[midi]
            b = ev
            best_by_midi[midi] = a if (note_velocity(a), note_duration(a)) >= (note_velocity(b), note_duration(b)) else b

    to_drop = set()

    for midi_high, ev_high in best_by_midi.items():
        v_high = note_velocity(ev_high)

        for interval in harmonic_intervals:
            midi_base = midi_high - int(interval)
            if midi_base in best_by_midi:
                ev_base = best_by_midi[midi_base]
                v_base = note_velocity(ev_base)

                if v_base >= v_high * min_base_velocity_ratio:
                    to_drop.add(midi_high)
                    break

    kept = [ev for ev in cluster if int(ev["midi_note"]) not in to_drop]
    return sort_by_onset(kept)


# ---------------------------------
# (A) Adaptive consistency filtering
# ---------------------------------
def adaptive_consistency_filter(
    note_events: list[dict],
    *,
    min_occurrences: int = 2,
    min_total_dur_ratio_of_max: float = 0.10,
    keep_if_velocity_ge: int | None = None,
) -> list[dict]:
    """
    Adaptive consistency filter across the whole audio.

    Goal:
      Remove "one-off" ghost notes that appear only once or have tiny total duration.

    How it works:
      - Count occurrences per MIDI note
      - Sum total duration per MIDI note
      - Keep notes that:
          (occurrences >= min_occurrences) OR
          (total_duration >= min_total_dur_ratio_of_max * max_total_duration)

      Optional override:
        keep_if_velocity_ge: if a note has velocity >= this threshold, keep it even if rare.
    """
    if not note_events:
        return note_events

    occ = defaultdict(int)
    tot_dur = defaultdict(float)

    for ev in note_events:
        midi = int(ev["midi_note"])
        occ[midi] += 1
        tot_dur[midi] += note_duration(ev)

    max_total = max(tot_dur.values()) if tot_dur else 0.0
    dur_threshold = (min_total_dur_ratio_of_max * max_total) if max_total > 0 else 0.0

    filtered = []
    for ev in note_events:
        midi = int(ev["midi_note"])
        v = note_velocity(ev)

        if keep_if_velocity_ge is not None and v >= int(keep_if_velocity_ge):
            filtered.append(ev)
            continue

        if occ[midi] >= int(min_occurrences):
            filtered.append(ev)
            continue

        if tot_dur[midi] >= float(dur_threshold):
            filtered.append(ev)
            continue

        # else: drop it
    return sort_by_onset(filtered)


# ---------------------------------
# Full filtering pipeline (A + B + D)
# ---------------------------------
def filter_note_events_ABD(
    note_events: list[dict],
    *,
    # (B) clustering params
    enable_B: bool = False,
    cluster_window: float = 0.04,
    dedupe_window: float = 0.08,
    max_notes_per_cluster: int = 6,
    # (D) harmonic params
    enable_D: bool = False,
    harmonic_velocity_ratio: float = 1.15,
    # (A) consistency params
    enable_A: bool = False,
    min_occurrences: int = 2,
    min_total_dur_ratio_of_max: float = 0.10,
) -> list[dict]:
    """
    Run the requested filters in a sensible order:

    1) (B) Cluster by onset, dedupe same pitch, dedupe pitch-class (octaves), keep top-K per chord moment,
       then global-dedupe same MIDI across clusters.
    2) (D) Drop likely harmonics inside each chord cluster
    3) (A) Adaptive consistency filter globally (remove one-off ghosts)

    You can enable/disable A/B/D independently.
    """
    events = sort_by_onset(note_events)

    # ---- (B) clustering filter ----
    if enable_B:
        clusters = cluster_by_onset(events, cluster_window=cluster_window)
        pruned: list[dict] = []

        for c in clusters:
            # 1) Deduplicate exact same MIDI inside the cluster
            c = dedupe_same_pitch_in_cluster(c, dedupe_window=dedupe_window)

            # 2) NEW: Remove octave duplicates (same pitch class) inside the cluster
            c = dedupe_pitch_class_in_cluster(c)

            # 3) Keep top-K strongest
            c = keep_top_k_in_cluster(c, max_notes=max_notes_per_cluster)

            pruned.extend(c)

        # 4) NEW: Dedupe same MIDI across nearby clusters globally
        events = dedupe_same_midi_globally(pruned, dedupe_window=dedupe_window)

    # ---- (D) harmonic filter ----
    if enable_D:
        clusters = cluster_by_onset(events, cluster_window=cluster_window)
        pruned: list[dict] = []
        for c in clusters:
            c = drop_likely_harmonics(c, min_base_velocity_ratio=harmonic_velocity_ratio)
            pruned.extend(c)
        events = sort_by_onset(pruned)

    # ---- (A) adaptive consistency filter ----
    if enable_A:
        events = adaptive_consistency_filter(
            events,
            min_occurrences=min_occurrences,
            min_total_dur_ratio_of_max=min_total_dur_ratio_of_max,
            keep_if_velocity_ge=None,
        )

    return events


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio into MIDI + note events (txt/json)."
    )

    # Core inputs
    parser.add_argument("--audio", required=True, help="Path to input audio file.")
    parser.add_argument("--outdir", default=None, help="Output directory (default: audio folder).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device.")
    parser.add_argument("--stem", default=None, help="Base filename for outputs (default: audio stem).")
    parser.add_argument("--no-midi", action="store_true", help="Do not keep the MIDI file.")
    parser.add_argument("--full-json", action="store_true", help="Write JSON-safe full result (large).")
    parser.add_argument("--print-raw", action="store_true", help="Also print the raw (unfiltered) notes.")
    parser.add_argument("--print-audio-info", action="store_true", help="Print audio duration and sample count.")

    # --- Filter toggles (A/B/D) ---
    parser.add_argument("--enable-A", action="store_true", help="Enable A: adaptive consistency filter.")
    parser.add_argument("--enable-B", action="store_true", help="Enable B: onset clustering filter.")
    parser.add_argument("--enable-D", action="store_true", help="Enable D: harmonic/overtone filter.")

    # --- B parameters ---
    parser.add_argument("--cluster-window", type=float, default=0.04, help="B: onset clustering window (sec).")
    parser.add_argument("--dedupe-window", type=float, default=0.08, help="B: same-pitch/global dedupe window (sec).")
    parser.add_argument("--max-notes-per-cluster", type=int, default=6, help="B: keep top-K per cluster.")

    # --- D parameters ---
    parser.add_argument(
        "--harmonic-velocity-ratio",
        type=float,
        default=1.15,
        help="D: base note must be >= (ratio * harmonic note velocity) to drop harmonic.",
    )

    # --- A parameters ---
    parser.add_argument("--min-occurrences", type=int, default=2, help="A: keep pitches appearing >= this many times.")
    parser.add_argument(
        "--min-total-dur-ratio-of-max",
        type=float,
        default=0.10,
        help="A: keep pitches whose total duration >= ratio * max_total_duration.",
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
    audio_dur = len(audio) / sample_rate

    if args.print_audio_info:
        print(f"Audio samples: {len(audio)}")
        print(f"Audio duration (s): {audio_dur:.3f}")

    # Transcribe
    transcriptor = PianoTranscription(device=args.device)
    result = transcriptor.transcribe(audio, str(out_mid))

    note_events_raw = result.get("est_note_events", [])
    pedal_events = result.get("est_pedal_events", [])

    # IMPORTANT FIX: clamp raw events to audio duration
    note_events_raw = clamp_events_to_audio(note_events_raw, audio_dur=audio_dur)

    # Optional: print raw notes
    if args.print_raw:
        print(build_notes_txt(note_events_raw, title="RAW notes (clamped to audio duration)"))

    # Apply ABD filters (any combination)
    note_events_filtered = filter_note_events_ABD(
        note_events_raw,
        enable_A=bool(args.enable_A),
        enable_B=bool(args.enable_B),
        enable_D=bool(args.enable_D),
        cluster_window=float(args.cluster_window),
        dedupe_window=float(args.dedupe_window),
        max_notes_per_cluster=int(args.max_notes_per_cluster),
        harmonic_velocity_ratio=float(args.harmonic_velocity_ratio),
        min_occurrences=int(args.min_occurrences),
        min_total_dur_ratio_of_max=float(args.min_total_dur_ratio_of_max),
    )

    # Print filtered notes
    print(build_notes_txt(note_events_filtered, title="Filtered notes"))

    # Pedal events
    if pedal_events:
        print("Pedal events (est_pedal_events):")
        for p in pedal_events:
            onset = float(p["onset_time"])
            offset = min(float(p["offset_time"]), audio_dur)  # clamp for display
            print(f"  onset={onset:.3f}s  offset={offset:.3f}s")

    # Save TXT
    out_txt.write_text(build_notes_txt(note_events_filtered, title="Filtered notes"), encoding="utf-8")

    # Save JSON
    if args.full_json:
        payload = make_json_safe(result)
    else:
        payload = {
            "audio_file": str(audio_path),
            "audio_duration_s": audio_dur,
            "note_events_raw": note_events_raw if args.print_raw else None,
            "note_events": note_events_filtered,
            "pedal_events": pedal_events,
            "filters": {
                "A_enabled": bool(args.enable_A),
                "B_enabled": bool(args.enable_B),
                "D_enabled": bool(args.enable_D),
                "cluster_window": args.cluster_window,
                "dedupe_window": args.dedupe_window,
                "max_notes_per_cluster": args.max_notes_per_cluster,
                "harmonic_velocity_ratio": args.harmonic_velocity_ratio,
                "min_occurrences": args.min_occurrences,
                "min_total_dur_ratio_of_max": args.min_total_dur_ratio_of_max,
            },
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

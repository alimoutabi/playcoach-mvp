#!/usr/bin/env python3
import json
from pathlib import Path
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{NOTE_NAMES[midi % 12]}{octave}"


def plot_ergebnisse(audio, noten, output_pfad):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    cqt = np.abs(librosa.cqt(audio, sr=sample_rate))
    librosa.display.specshow(librosa.power_to_db(cqt**2, ref=np.max), sr=sample_rate, x_axis='time', y_axis='cqt_note')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Frequenz-Analyse von Input Audio')
    plt.subplot(2, 1, 2)
    for n in noten:
        start = float(n["onset_time"])
        end = float(n["offset_time"])
        midi = int(n["midi_note"])
        plt.plot([start, end], [midi, midi], color='green', linewidth=3)
    
    plt.xlabel('Zeit (s)')
    plt.ylabel('MIDI Note')
    plt.title('Erkannte Noten (Piano Roll)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_pfad)
    print(f"Visualisierung gespeichert unter: {output_pfad}")
    plt.show()



def main():
    audio_file = Path("C:/Users/Farid/Chorderkennung/A.m4a")  
    Output_dir = Path("C:/Users/Farid/Chorderkennung/output")
    Output_name = "output" 
    Device_name = "cpu"  
    bild_pfad = Output_dir / f"{Output_name}_plot.png"
    Output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Lade Audio: {audio_file}...")
    audio, _ = librosa.load(str(audio_file), sr=sample_rate, mono=True)
    print("Analysiere Klavierspiel (das kann kurz dauern)...")
    transcriptor = PianoTranscription(device=Device_name)
    midi_dir = Output_dir / f"{Output_name}.mid"
    result = transcriptor.transcribe(audio, str(midi_dir))
    noten = result.get("est_note_events", [])
    text_output = "Nr\tMIDI\tName\tStart(s)\tEnde(s)\tStärke\n"
    for i, n in enumerate(noten):
        start = float(n["onset_time"])
        end = float(n["offset_time"])
        midi = int(n["midi_note"])
        vel = n.get("velocity", 0)
        name = midi_to_name(midi)
        text_output += f"{i}\t{midi}\t{name}\t{start:.2f}\t{end:.2f}\t{vel}\n"

    txt_pfad = Output_dir / f"{Output_name}_noten.txt"
    txt_pfad.write_text(text_output, encoding="utf-8")
    json_pfad = Output_dir / f"{Output_name}_daten.json"
    with open(json_pfad, "w", encoding="utf-8") as f:
        json.dump(noten, f, indent=2, default=float)

    plot_ergebnisse(audio, noten, bild_pfad)
    print("\nFertig!")
    print(f"Gespeichert in: {Output_dir}")
    print(f"Dateien: {Output_name}.mid, {Output_name}_noten.txt, {Output_name}_daten.json")

main()
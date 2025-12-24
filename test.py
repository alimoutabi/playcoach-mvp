import librosa
from piano_transcription_inference import PianoTranscription, sample_rate

audio_path = r"/Users/alimoutabi/Desktop/notes/A.m4a"   # make sure this file exists

audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

transcriptor = PianoTranscription(device="cpu")  # or "cuda"
result = transcriptor.transcribe(audio, "/Users/alimoutabi/Desktop/notes/out.mid")

print(result.keys())

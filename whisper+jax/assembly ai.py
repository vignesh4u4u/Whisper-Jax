import assemblyai as aai
aai.settings.api_key = "13d2eb1f119640c185c2f5c200e10e22"
FILE_URL = r"C:\Users\VigneshSubramani\Music\sample_audio\Hitch_1_MS-POV.mp3"
config = aai.TranscriptionConfig(speaker_labels=True)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(
  FILE_URL,
  config=config
)
for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")

from whisper_jax import FlaxWhisperPipline
pipeline = FlaxWhisperPipline("openai/whisper-tiny")
path = r"C:\Users\VigneshSubramani\Music\sample_audio\sample1.flac"
outputs = pipeline(path,  task="translate", return_timestamps=True)
text = outputs["text"]
chunks = outputs["chunks"]
print(text)
print(chunks)
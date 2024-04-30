#https://picovoice.ai/blog/speaker-diarization-in-python/

from gradio_client import Client


API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"


client = Client(API_URL)


def transcribe_audio(audio_path, task="transcribe", return_timestamps=False):
    """Function to transcribe an audio file using the Whisper JAX endpoint."""
    if task not in ["transcribe", "translate"]:
        raise ValueError("task should be one of 'transcribe' or 'translate'.")

    text, runtime = client.predict(
        audio_path,
        task,
        return_timestamps,
        api_name="/predict_1",
    )
    return text

audio_path= r"C:\Users\VigneshSubramani\Music\sample_audio\Hitch_1.mp3"
output = transcribe_audio(audio_path)
output_with_timestamps = transcribe_audio(audio_path, return_timestamps=True)

print(output)
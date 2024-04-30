import gc
import tempfile
import os

import whisper
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Pipeline
from audio_video_speech_recognition.diarization_model import load_model

pipeline = load_model()


def read(k):
    y = np.array(k.get_array_of_samples())
    return np.float32(y) / 32768


def start_time(start_time_str):
    start_time_inte = int(start_time_str[:2]) * 3600000 + int(start_time_str[3:5]) * 60000 + int(
        start_time_str[6:8]) * 1000 + int(start_time_str[9:])
    return start_time_inte


def end_time(end_time_str):
    end_time_inte = int(end_time_str[:2]) * 3600000 + int(end_time_str[3:5]) * 60000 + int(
        end_time_str[6:8]) * 1000 + int(
        end_time_str[9:])
    return end_time_inte


def convert_audio_to_text(file):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        temp_audio_path = temp_audio_file.name
        file.save(temp_audio_path)

    k = str(pipeline(temp_audio_path)).split('\n')
    audio = AudioSegment.from_file(temp_audio_path)
    audio = audio.set_frame_rate(16000)
    model = whisper.load_model("base")
    transcription_output = []

    for line in k:
        j = line.split(" ")
        start_time_str = j[1]
        end_time_str = j[4].rstrip("]")
        st = start_time(start_time_str)
        et = end_time(end_time_str)
        tr = read(audio[st:et])
        result = model.transcribe(tr, fp16=False)
        speaker = j[6]
        text = result["text"]
        transcription_output.append((speaker, text))

    with open("transcription_output.txt", "w") as file:
        for speaker, text in transcription_output:
            file.write(f"[{start_time_str} -- {end_time_str}] {speaker}: {text}\n")

    with open("transcription_output.txt", "r") as file:
        saved_transcription = file.read()
    os.remove("transcription_output.txt")
    os.remove(temp_audio_path)

    return saved_transcription
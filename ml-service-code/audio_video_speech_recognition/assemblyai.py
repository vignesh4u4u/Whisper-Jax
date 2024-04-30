from flask import Flask, request
import os
import cv2
from moviepy.editor import AudioFileClip
import whisper
model = whisper.load_model("base")
import assemblyai as aai

def transcribe_audio(file_url):
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        file_url,
        config=config
    )
    transcriptions = []
    for utterance in transcript.utterances:
        transcriptions.append((utterance.speaker, utterance.text))
    return transcriptions

def extract_text_binary_files(input_file):
    if input_file.filename == '':
        return "No selected file"
    if input_file.filename.endswith((".mp4", ".avi", ".mov")):
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        video_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(video_path)
        cap = cv2.VideoCapture(video_path)
        audio_path = os.path.join(temp_dir, "audio.wav")
        clip = AudioFileClip(video_path)
        clip.write_audiofile(audio_path)
        FILE_URL = audio_path
        transcription_output = transcribe_audio(FILE_URL)
        with open("transcription_output.txt", "w") as file:
            for speaker, text in transcription_output:
                file.write(f"Speaker {speaker}: {text}\n")

        with open("transcription_output.txt", "r") as file:
            saved_transcription = file.read()
        return saved_transcription
        os.remove("transcription_output.txt")
        os.remove(audio_path)
        cap.release()
        os.remove(video_path)

    elif input_file.filename.endswith((".mp3", ".wav", ".ogg", ".flac")):
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        audio_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(audio_path)
        FILE_URL = audio_path
        FILE_URL = audio_path
        transcription_output = transcribe_audio(FILE_URL)
        with open("transcription_output.txt", "w") as file:
            for speaker, text in transcription_output:
                file.write(f"Speaker {speaker}: {text}\n")

        with open("transcription_output.txt", "r") as file:
            saved_transcription = file.read()
        return saved_transcription
        os.remove("transcription_output.txt")
        os.remove(audio_path)
from flask import Flask, request
import os
import cv2
from moviepy.editor import AudioFileClip
import whisper
model = whisper.load_model("base")

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
        result = model.transcribe(audio_path)
        os.remove(audio_path)
        cap.release()
        os.remove(video_path)
        return {"voice_text ": result["text"]}
    elif input_file.filename.endswith((".mp3", ".wav", ".ogg", ".flac")):
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        audio_path = os.path.join(temp_dir, input_file.filename)
        input_file.save(audio_path)
        result = model.transcribe(audio_path)
        os.remove(audio_path)
        return {"voice_text ": result["text"]}

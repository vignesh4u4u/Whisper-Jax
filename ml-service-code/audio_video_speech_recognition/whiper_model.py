from flask import Flask, request
import os
import cv2
from moviepy.editor import AudioFileClip
import whisper
model = whisper.load_model("base")
app = Flask(__name__)

@app.route("/video_to_audio", methods=["POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        input_file = request.files["file"]

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
            return {"voice_text ":result["text"]}
        elif input_file.filename.endswith((".mp3", ".wav", ".ogg",".flac")):
            temp_dir = "temp"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            audio_path = os.path.join(temp_dir, input_file.filename)
            input_file.save(audio_path)
            result = model.transcribe(audio_path)
            os.remove(audio_path)
            return {"voice_text ": result["text"]}

if __name__ == "__main__":
    app.run(debug=True)




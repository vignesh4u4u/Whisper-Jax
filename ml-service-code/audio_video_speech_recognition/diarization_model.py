from pyannote.audio import Pipeline

def load_model():
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token="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")
    return pipeline
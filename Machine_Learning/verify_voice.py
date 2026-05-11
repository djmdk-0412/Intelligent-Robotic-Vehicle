import joblib
import librosa
import numpy as np
import warnings

# This keeps the LabVIEW window clean from unimportant warnings
warnings.filterwarnings('ignore')

def verify_voice(audio_file_path):
    try:
        # 1. Path to your freshly trained brain
        model_path = r"C:\Users\Asus\Documents\Sentry_Project\voice_lock_model.pkl"
        model = joblib.load(model_path)
        
        # 2. Process the test audio
        # Using 2 seconds to match the training data
        y, sr = librosa.load(audio_file_path, duration=2.0)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0).reshape(1, -1)
        
        # 3. Get the prediction
        # Result will be 1 (Boss) or 0 (Stranger)
        prediction = model.predict(features)[0]
        return int(prediction)
        
    except Exception as e:
        # If something goes wrong, return -1 so we know it crashed
        return -1
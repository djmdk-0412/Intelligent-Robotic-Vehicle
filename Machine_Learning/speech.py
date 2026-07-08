import pyttsx3
import pyaudio
import json
import os
import wave
import librosa
import numpy as np
import joblib
from vosk import Model, KaldiRecognizer

# ==========================================
# 1. LOAD THE BRAINS (Vosk & Random Forest)
# ==========================================
# Automatically get the exact directory where this speech.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stick the base directory to the front of the file names
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-en-us-0.15")
PKL_PATH = os.path.join(BASE_DIR, "voice_model.pkl")

if os.path.exists(MODEL_PATH):
    vosk_model = Model(MODEL_PATH)
else:
    vosk_model = None

# Load your custom voice signature model
if os.path.exists(PKL_PATH):
    ml_model = joblib.load(PKL_PATH)
else:
    ml_model = None

def listen_and_control():
    # Initialize the TTS Engine (Speaker Voice)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 185)

    # Failsafe: Check if models are missing
    if vosk_model is None or ml_model is None:
        engine.say("System files missing.")
        engine.runAndWait()
        return "Error: Models not found"

    try:
        # Added "self" to the vocabulary list to ensure "self parking" is captured
        rec = KaldiRecognizer(vosk_model, 16000, '["forward", "backward", "left", "right", "stop", "self", "parking", "intruder", "shutdown", "exit", "[unk]"]')
        
        # Open the Microphone
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4000)
        stream.start_stream()

        command = ""
        frames = [] # We will save the audio frames here for the ML model

        # Listen in real-time
        for _ in range(0, 40):
            data = stream.read(4000, exception_on_overflow=False)
            frames.append(data) 
            
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                command = result.get("text", "")
                if command:
                    # WE HEARD THE WORD! Record for 0.5 more seconds to get the full voice signature
                    for _ in range(5):
                        extra_data = stream.read(4000, exception_on_overflow=False)
                        frames.append(extra_data)
                    break 

        # Cleanly shut down the microphone
        stream.stop_stream()
        stream.close()
        p.terminate()

        if command == "":
            return "No command recognized"

        # ==========================================
        # 2. VERIFY THE VOICE (Extract Features)
        # ==========================================
        temp_wav = os.path.join(BASE_DIR, "temp_live.wav")
        
        # FORCE DELETE the old file so we don't accidentally read old audio!
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                pass # If it's locked, just move on and overwrite
            
        # Save the freshly captured audio
        wf = wave.open(temp_wav, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Do the MFCC Math (Audio -> Numbers)
        audio, sample_rate = librosa.load(temp_wav, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_averaged = np.mean(mfccs.T, axis=0)

        # Make the prediction using CONFIDENCE LEVELS!
        probabilities = ml_model.predict_proba([mfccs_averaged])[0]
        
        # Find exactly which percentage belongs to the Owner (Class 1)
        owner_index = np.where(ml_model.classes_ == 1)[0][0]
        owner_confidence = probabilities[owner_index]

        # --- NEW: Print the exact math to the console so we can see what is happening! ---
        print(f"\n--- DEBUG: Owner Confidence Score = {owner_confidence:.3f} ---\n")

        # ==========================================
        # 3. ROUTE TO LABVIEW
        # ==========================================
        # HIGH SECURITY: The AI must be at least 75% confident it is the Owner
        if owner_confidence < 0.75:
            # INTRUDER DETECTED!
            engine.say("Access Denied. Intruder detected.")
            engine.runAndWait()
            return "Access Denied" 

        else:
            # OWNER VERIFIED! Map the specific command actions.
            if "forward" in command:
                action, spoken = "Moving forward", "Move forward"
            elif "backward" in command:
                action, spoken = "Moving backward", "Move backward"
            elif "left" in command:
                action, spoken = "Turning left", "Turn left"
            elif "right" in command:
                action, spoken = "Turning right", "Turn right"
            elif "stop" in command:
                action, spoken = "Stopping", "Stop"
            elif "parking" in command or "self" in command:
                action, spoken = "Self parking mode activated", "Self parking"
            elif "shutdown" in command or "exit" in command:
                action, spoken = "Shutting down", "Shutting down"
            else:
                action, spoken = f"Unrecognized command: {command}", "Command not recognized"

            engine.say(spoken)
            engine.runAndWait()
            return action

    except Exception as e:
        return "System error"


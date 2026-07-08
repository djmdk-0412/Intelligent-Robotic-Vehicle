import pyaudio
import wave
import librosa
import numpy as np
import joblib
import time
import os

# ==========================================
# 1. LOAD YOUR TRAINED AI MODEL
# ==========================================
print("Loading voice model...")
if not os.path.exists('voice_model.pkl'):
    print("❌ Error: 'voice_model.pkl' not found! Make sure train_model.py finished successfully.")
    exit()
    
model = joblib.load('voice_model.pkl')

# ==========================================
# 2. RECORD LIVE AUDIO (3 SECONDS)
# ==========================================
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050 
# 🔥 Increased from 2 to 3 seconds so you don't feel rushed!
RECORD_SECONDS = 3 
WAVE_OUTPUT_FILENAME = "temp_test.wav"

p = pyaudio.PyAudio()

print("\n🎤 GET READY... Speak a command in:")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")

# 🔥 Added a tiny 0.5-second hardware delay here 
time.sleep(0.5) 
print("\n🔴 RECORDING NOW! (Speak!)")

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("✅ Recording stopped.\n")

stream.stop_stream()
stream.close()
p.terminate()

# Save the recording to a temporary file
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# ==========================================
# 3. DO THE AUDIO MATH (MFCC)
# ==========================================
print("Analyzing vocal signature...")
try:
    audio, sample_rate = librosa.load(WAVE_OUTPUT_FILENAME, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_averaged = np.mean(mfccs.T, axis=0)
except Exception as e:
    print("Error analyzing audio:", e)
    exit()

# ==========================================
# 4. MAKE THE CLASSIFICATION GUESS!
# ==========================================
# We wrap the numbers in brackets [ ] because sklearn expects a 2D array
prediction = model.predict([mfccs_averaged])

print("\n" + "="*40)
if prediction[0] == 1:
    print("🟢 RESULT: OWNER VERIFIED! Welcome back.")
else:
    print("🔴 RESULT: INTRUDER DETECTED! Access Denied.")
print("="*40 + "\n")
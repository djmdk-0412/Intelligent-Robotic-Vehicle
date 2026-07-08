import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# ==========================================
# 1. THE MFCC EXTRACTOR (Turning sound into numbers)
# ==========================================
def extract_features(file_path):
    try:
        # Load the audio file using Librosa
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Calculate the MFCCs (Extracting 40 distinct voice features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Average the numbers out so every audio file gives us exactly 40 numbers, 
        # no matter how long the recording is!
        mfccs_averaged = np.mean(mfccs.T, axis=0)
        
        return mfccs_averaged
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# ==========================================
# 2. LOAD THE DATASET
# ==========================================
features = [] # This will hold the MFCC numbers
labels = []   # This will hold the answers (1 for Owner, 0 for Intruder)

print("Extracting features from the 'owner' folder...")
owner_path = "dataset/owner"
for file in os.listdir(owner_path):
    if file.endswith(".wav"):
        data = extract_features(os.path.join(owner_path, file))
        if data is not None:
            features.append(data)
            labels.append(1) # 1 means "Owner"

print("Extracting features from the 'intruder' folder...")
intruder_path = "dataset/intruder"
for file in os.listdir(intruder_path):
    if file.endswith(".wav"):
        data = extract_features(os.path.join(intruder_path, file))
        if data is not None:
            features.append(data)
            labels.append(0) # 0 means "Intruder"

# ==========================================
# 3. TRAIN THE RANDOM FOREST ALGORITHM
# ==========================================
print("Training the Random Forest Classifier...")

# Convert our lists to numpy arrays (math tables that sklearn can read)
X = np.array(features)
y = np.array(labels)

# Build the Random Forest with 100 Decision Trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# TRAIN IT! (This tells the math to find the boundaries between the 1s and 0s)
model.fit(X, y)

# ==========================================
# 4. SAVE THE BRAIN FOR LABVIEW TO USE
# ==========================================
joblib.dump(model, 'voice_model.pkl')
print("✅ Success! 'voice_model.pkl' has been saved. You can now use it in your main code.")
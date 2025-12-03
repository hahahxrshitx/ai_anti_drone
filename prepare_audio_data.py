import os
import librosa
import numpy as np
from tqdm import tqdm

# -------------------------------
# CONFIG: Update these paths
# -------------------------------

# Multiclass paths (3-class CNN)
multiclass_paths = {
    "drone_hover": r"C:\AI_Anti_Drone\datasets\audio\DroneAudioDataset\Multiclass_Drone_Audio\drone_hover",
    "drone_transit": r"C:\AI_Anti_Drone\datasets\audio\DroneAudioDataset\Multiclass_Drone_Audio\drone_transit",
    "background_noise": r"C:\AI_Anti_Drone\datasets\audio\DroneAudioDataset\Multiclass_Drone_Audio\bg_noise"
}

# Output folder for spectrograms
output_dir = r"C:\AI_Anti_Drone\datasets\audio\spectrograms"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# PARAMETERS
# -------------------------------
sample_rate = 16000       # Resample all audio to 16 kHz
clip_duration = 3         # Duration in seconds
n_mfcc = 40               # Number of MFCC features

# -------------------------------
# FUNCTION: Convert audio to spectrogram
# -------------------------------
def process_audio_folder(folder_path, class_name, output_base):
    output_class_dir = os.path.join(output_base, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    
    for file in tqdm(files, desc=f"Processing {class_name}"):
        file_path = os.path.join(folder_path, file)
        # Load and resample
        y, sr = librosa.load(file_path, sr=sample_rate)
        
        # Trim or pad
        desired_length = clip_duration * sample_rate
        if len(y) < desired_length:
            y = np.pad(y, (0, desired_length - len(y)))
        else:
            y = y[:desired_length]

        # Convert to MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Save as numpy file (.npy)
        npy_filename = os.path.join(output_class_dir, os.path.splitext(file)[0] + ".npy")
        np.save(npy_filename, mfcc)

# -------------------------------
# PROCESS MULTICLASS DATA ONLY
# -------------------------------
print("=== Processing Multiclass Audio ===")
for class_name, folder_path in multiclass_paths.items():
    process_audio_folder(folder_path, class_name, output_dir)

print("All audio processed successfully! Spectrograms saved in:", output_dir)

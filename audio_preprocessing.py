import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import librosa

# -------------------------------
# CONFIG: Update these paths
# -------------------------------
spectrogram_dir = r"C:\AI_Anti_Drone\datasets\audio\spectrograms"
output_dir = r"C:\AI_Anti_Drone\datasets\audio\preprocessed"
os.makedirs(output_dir, exist_ok=True)

# Train/Val/Test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Optional augmentation flags
apply_augmentation = True

# -------------------------------
# FUNCTIONS
# -------------------------------

def normalize_mfcc(mfcc):
    return (mfcc - np.mean(mfcc)) / np.std(mfcc)

def add_channel_dim(mfcc):
    return np.expand_dims(mfcc, axis=-1)  # shape -> (n_mfcc, n_frames, 1)

def augment_mfcc(y, sr):
    # Simple augmentation: time stretch or pitch shift
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.9, 1.1))
    if np.random.rand() < 0.5:
        y = librosa.effects.pitch_shift(y, sr, n_steps=np.random.randint(-2, 3))
    return y

def process_class(class_name, file_list):
    X, y = [], []
    for file_path in tqdm(file_list, desc=f"Processing {class_name}"):
        mfcc = np.load(file_path)

        # Normalize
        mfcc = normalize_mfcc(mfcc)

        # Add channel dimension
        mfcc = add_channel_dim(mfcc)

        X.append(mfcc)
        y.append(class_name)
    return X, y

# -------------------------------
# LOAD FILES AND SPLIT DATA
# -------------------------------
all_classes = [d for d in os.listdir(spectrogram_dir) if os.path.isdir(os.path.join(spectrogram_dir, d))]

X_all, y_all = [], []

for cls in all_classes:
    cls_path = os.path.join(spectrogram_dir, cls)
    files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith(".npy")]
    
    X_cls, y_cls = process_class(cls, files)
    X_all.extend(X_cls)
    y_all.extend(y_cls)

# Convert labels to numeric
class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
y_all_num = np.array([class_to_idx[label] for label in y_all])

# Convert X_all to numpy array
X_all = np.array(X_all)

# Train / val / test split
X_temp, X_test, y_temp, y_test = train_test_split(X_all, y_all_num, test_size=test_ratio, stratify=y_all_num, random_state=42)
val_size_adjusted = val_ratio / (train_ratio + val_ratio)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=42)

# -------------------------------
# SAVE PREPROCESSED DATA
# -------------------------------
np.save(os.path.join(output_dir, "X_train.npy"), X_train)
np.save(os.path.join(output_dir, "y_train.npy"), y_train)
np.save(os.path.join(output_dir, "X_val.npy"), X_val)
np.save(os.path.join(output_dir, "y_val.npy"), y_val)
np.save(os.path.join(output_dir, "X_test.npy"), X_test)
np.save(os.path.join(output_dir, "y_test.npy"), y_test)

print("Preprocessing complete! Data saved to:", output_dir)
print("Classes mapping:", class_to_idx)

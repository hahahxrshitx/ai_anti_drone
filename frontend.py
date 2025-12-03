import cv2
import sounddevice as sd
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from scipy.io.wavfile import write
from datetime import datetime
import tkinter as tk
from tkinter import scrolledtext
import csv

# === STEP 1: Audio Preprocessing ===
from prepare_audio_data import audio_to_spectrogram

# === STEP 3: Threat & Intent Engine ===
from threat_engine import classify_flight_pattern, assess_threat

# === STEP 2: Load Trained Models ===
yolo_model = YOLO('trained_models/best.pt')
cnn_model = tf.keras.models.load_model('trained_models/acoustic_model.h5')

# === Initialize Webcam Stream ===
cap = cv2.VideoCapture(0)
frame_buffer = []

# === CSV Logging Setup ===
csv_file = open("threat_log.csv", "a", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Visual", "Acoustic", "Behavior", "Assessment"])

def log_threat_csv(timestamp, visual, acoustic, behavior, assessment):
    csv_writer.writerow([timestamp, visual, acoustic, behavior, assessment])
    csv_file.flush()

# === Tkinter GUI Setup ===
root = tk.Tk()
root.title("Drone Surveillance Dashboard")

status_box = scrolledtext.ScrolledText(root, width=70, height=20, font=("Consolas", 10))
status_box.pack()

def update_gui(visual, acoustic, behavior, assessment, timestamp):
    status_box.insert(tk.END, f"[{timestamp}] VISUAL={visual}, ACOUSTIC={acoustic}, BEHAVIOR={behavior}, ASSESSMENT={assessment}\n")
    status_box.see(tk.END)
    root.update()

# === Microphone Recording Function ===
def record_audio(duration=3, fs=44100):
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write("temp.wav", fs, audio)
    return "temp.wav"

# === Main Surveillance Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam stream failed.")
        break

    # --- YOLOv8 Detection ---
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    centers = [(int((x1+x2)/2), int((y1+y2)/2)) for x1, y1, x2, y2 in boxes]
    frame_buffer.append(centers)
    if len(frame_buffer) > 30:
        frame_buffer.pop(0)

    # --- Acoustic Detection ---
    audio_path = record_audio()
    spec_img = audio_to_spectrogram(audio_path)
    spec_img = np.expand_dims(spec_img, axis=0)
    acoustic_pred = cnn_model.predict(spec_img, verbose=0)
    acoustic_label = np.argmax(acoustic_pred)
    acoustic_map = {0: "NOISE", 1: "HOVER", 2: "TRANSIT"}
    acoustic_status = acoustic_map.get(acoustic_label, "UNKNOWN")

    # --- Flight Pattern Classification ---
    behavior = classify_flight_pattern(frame_buffer)

    # --- Threat Assessment ---
    final_status = assess_threat(
        visual_detected=len(boxes) > 0,
        acoustic_label=acoustic_label,
        behavior=behavior
    )

    # --- Logging + GUI Update ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_threat_csv(timestamp, "DETECTED" if boxes else "NONE", acoustic_status, behavior, final_status)
    update_gui("DETECTED" if boxes else "NONE", acoustic_status, behavior, final_status, timestamp)

    # --- Overlay Results on Video ---
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(frame, f"VISUAL: {'DETECTED' if boxes else 'NONE'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"ACOUSTIC: {acoustic_status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"BEHAVIOR: {behavior}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"ASSESSMENT: {final_status}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"TIME: {timestamp}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    cv2.imshow("Surveillance Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
csv_file.close()

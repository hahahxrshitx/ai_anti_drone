from audiomentations import Compose, AddGaussianNoise, Gain, PitchShift
import soundfile as sf
import os

audio_base = 'datasets/audio/DroneAudioDataset'
audio_classes = [d for d in os.listdir(audio_base) if os.path.isdir(os.path.join(audio_base, d))]
attacks = ['noise', 'quiet', 'pitch']

for cls in audio_classes:
    cls_path = os.path.join(audio_base, cls)
    wav_files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
    for attack in attacks:
        os.makedirs(f'adversarial_audio/{cls}_{attack}', exist_ok=True)
    for wav_file in wav_files:
        samples, sr = sf.read(os.path.join(cls_path, wav_file))
        # Noise
        noisy = AddGaussianNoise(0.01, 0.05, p=1.0)(samples, sr)
        sf.write(f'adversarial_audio/{cls}_noise/{wav_file}', noisy, sr)
        # Quiet
        quiet = Gain(-15, -5, p=1.0)(samples, sr)
        sf.write(f'adversarial_audio/{cls}_quiet/{wav_file}', quiet, sr)
        # Pitch shift
        pitched = PitchShift(-2, 2, p=1.0)(samples, sr)
        sf.write(f'adversarial_audio/{cls}_pitch/{wav_file}', pitched, sr)
print("Adversarial audio generation complete.")
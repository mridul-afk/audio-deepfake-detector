# from pathlib import Path
# import torch
# import torch.nn as nn

# class AudioDeepfakeDetector(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1))
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),         # Index 0
#             nn.Dropout(0.3),      # Index 1
#             nn.Linear(128, 64),   # Index 2
#             nn.ReLU(),            # Index 3
#             nn.Dropout(0.3),      # Index 4
#             nn.Linear(64, 2)      # Index 5
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# # Load and evaluate
# model_path = Path("models/best_deepfake_detector.pth")
# model = AudioDeepfakeDetector()

# state_dict = torch.load(model_path, map_location="cpu")
# model.load_state_dict(state_dict)
# model.eval()

# print(f"Model successfully loaded from: {model_path.resolve()}\n")

# # Run dummy test pass with a 128x128 audio feature representation
# dummy_input = torch.randn(1, 1, 128, 128)

# with torch.no_grad():
#     logits = model(dummy_input)
#     probs = torch.softmax(logits, dim=1)

# real_score = probs[0][0].item()
# fake_score = probs[0][1].item()
# prediction = "Fake / Synthetic" if fake_score > real_score else "Real / Authentic"

# print("Test Pass Successful!")
# print(f" - Raw Logits: {logits.numpy()}")
# print(f" - Real Score: {real_score:.2%}")
# print(f" - Fake Score: {fake_score:.2%}")
# print(f" - Classification: {prediction}")





from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

# 1. Define Model Architecture
class AudioDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# 2. Audio Preprocessing Pipeline
def process_audio(audio_path, target_sr=16000, n_mels=128):
    waveform, sr = torchaudio.load(audio_path)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Generate Mel-Spectrogram in dB scale
    mel_transform = T.MelSpectrogram(sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=n_mels)
    mel_spec = T.AmplitudeToDB()(mel_transform(waveform))

    # Format to shape [batch=1, channel=1, n_mels, time]
    return mel_spec.unsqueeze(0)

# 3. Load Model Weights
model_path = Path("models/best_deepfake_detector.pth")
model = AudioDeepfakeDetector()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# 4. Run Prediction on File
audio_file = "sample.wav"  # Replace with your test audio file path

try:
    input_tensor = process_audio(audio_file)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    real_prob, fake_prob = probs[0][0].item(), probs[0][1].item()
    result = "Fake / Synthetic" if fake_prob > real_prob else "Real / Authentic"

    print(f"File Evaluated: {audio_file}")
    print(f" - Real Score: {real_prob:.2%}")
    print(f" - Fake Score: {fake_prob:.2%}")
    print(f" - Final Result: {result}")

except FileNotFoundError:
    print(f"Please provide a valid audio path for '{audio_file}' to perform evaluation.")

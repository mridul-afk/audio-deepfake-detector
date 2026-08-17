"""
backend/preprocess.py

The Digital Signal Processing (DSP) stage of the pipeline.

Job: turn a raw audio file (.flac/.wav/.mp3) into a fixed-size,
normalized log-Mel spectrogram — a 2D "image" that the CNN in model.py
can classify as bonafide (real) or spoof (fake).

Used from two places:
  1. prepare_dataset.py's CSV registry -> a Dataset class (see
     ASVspoofDataset below) during training.
  2. main.py's /predict endpoint, on a single uploaded file, at
     inference time.

Both paths MUST use the exact same preprocess_audio() function, or the
model will be trained on features that don't match what it sees in
production.
"""

import numpy as np
import librosa

# --------------------------------------------------------------------
# Configuration — kept as module-level constants so training and
# inference can never accidentally drift apart.
# --------------------------------------------------------------------
SAMPLE_RATE = 16000          # ASVspoof audio is natively 16 kHz
DURATION_SECONDS = 4         # fixed clip length fed to the CNN
N_SAMPLES = SAMPLE_RATE * DURATION_SECONDS

N_FFT = 1024                 # window size for each FFT frame
HOP_LENGTH = 256              # stride between frames (75% overlap)
N_MELS = 128                  # number of Mel filterbank bands


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file as mono and resample to `sample_rate`."""
    signal, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return signal


def pad_or_trim(signal: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Force every clip to the same length so the CNN gets a constant
    input shape, regardless of how long the original recording was.

    - Clips longer than n_samples are trimmed to the first N_SAMPLES.
    - Clips shorter are zero-padded (silence) at the end.
    """
    if len(signal) >= n_samples:
        return signal[:n_samples]
    return np.pad(signal, (0, n_samples - len(signal)), mode="constant")


def audio_to_melspectrogram(signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Convert a 1D waveform into a log-scaled Mel-spectrogram.

    Why Mel + log-dB, specifically for deepfake detection:
    - The Mel scale emphasizes the frequency bands the human vocal
      tract actually produces, which is exactly where vocoder-based
      TTS/voice-clone artifacts tend to leak in as unnatural texture.
    - Raw power spectrograms have a huge dynamic range; converting to
      decibels compresses that range so the CNN can learn from quiet
      artifacts (like phase glitches) as easily as loud ones.

    Returns a 2D array of shape (N_MELS, time_frames).
    """
    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    return log_mel_spec


def normalize(spec: np.ndarray) -> np.ndarray:
    """Min-max scale the spectrogram to [0, 1].

    Neural nets train far more reliably on small, consistently-scaled
    inputs than on raw dB values (which can range roughly -80 to 0).
    """
    spec_min, spec_max = spec.min(), spec.max()
    if spec_max - spec_min < 1e-6:
        # Degenerate case: pure silence / constant signal.
        return np.zeros_like(spec)
    return (spec - spec_min) / (spec_max - spec_min)


def preprocess_audio(audio_path: str) -> np.ndarray:
    """Full DSP pipeline: file path -> model-ready tensor.

    This is the single function main.py and the training Dataset
    should both call, so inference always matches training exactly.

    Returns a float32 array of shape (1, N_MELS, time_frames) —
    the leading 1 is the channel dimension PyTorch's Conv2d expects
    (batch, channels, height, width), with batch added separately
    by the DataLoader / by main.py before the forward pass.
    """
    signal = load_audio(audio_path)
    signal = pad_or_trim(signal)
    spec = audio_to_melspectrogram(signal)
    spec = normalize(spec)
    spec = np.expand_dims(spec, axis=0)  # (H, W) -> (1, H, W)
    return spec.astype(np.float32)


# --------------------------------------------------------------------
# Optional: PyTorch Dataset wired directly to prepare_dataset.py's
# CSV registry, so training just becomes:
#     from preprocess import ASVspoofDataset
#     train_ds = ASVspoofDataset("data_registry/train_metadata.csv")
# --------------------------------------------------------------------
try:
    import pandas as pd
    import torch
    from torch.utils.data import Dataset

    class ASVspoofDataset(Dataset):
        """Reads the CSV built by prepare_dataset.py and applies the
        DSP pipeline on the fly, per-sample."""

        def __init__(self, csv_path: str):
            self.df = pd.read_csv(csv_path)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            features = preprocess_audio(row["audio_path"])
            label = torch.tensor(row["label"], dtype=torch.long)
            return torch.from_numpy(features), label

except ImportError:
    # pandas/torch aren't required just to run the DSP functions above
    # (e.g. inside main.py's inference path if you keep them separate).
    pass


if __name__ == "__main__":
    # Quick manual sanity check:
    #   python preprocess.py path/to/some_clip.flac
    import sys

    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <path_to_audio_file>")
        sys.exit(1)

    features = preprocess_audio(sys.argv[1])
    print(f"✅ Processed '{sys.argv[1]}'")
    print(f"   Output shape: {features.shape}  (channels, n_mels, time_frames)")
    print(f"   Value range:  [{features.min():.3f}, {features.max():.3f}]")

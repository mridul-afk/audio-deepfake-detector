import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import dataset loader and model architecture
from preprocess import ASVspoofDataset
from model import DeepfakeAudioCNN


def run_epoch(model, dataloader, criterion, optimizer, device, is_training=True):
    """Executes a single epoch for either training or validation."""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Disable gradient tracking during validation
    with torch.set_grad_enabled(is_training):
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            if is_training:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            # Calculate accuracy metrics
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100.0
    return avg_loss, accuracy


def main():
    # 1. Hardware Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Training target device: {device}")

    # 2. File Paths & Hyperparameters
    BASE_DIR = Path(__file__).resolve().parent.parent

    train_csv = str(BASE_DIR / "data_registry" / "train_metadata.csv")
    dev_csv = str(BASE_DIR / "data_registry" / "dev_metadata.csv")

    checkpoint_dir = Path(__file__).resolve().parent / "models"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = str(checkpoint_dir / "best_deepfake_detector.pth")

    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    epochs = 15
    num_workers = 0  # Kept at 0 to avoid Windows DataLoader multiprocessing errors

    # 3. Datasets & DataLoaders
    print("📦 Loading Train and Validation Datasets...")
    train_dataset = ASVspoofDataset(train_csv)
    dev_dataset = ASVspoofDataset(dev_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Dev validation samples: {len(dev_dataset)}")

    # 4. Model, Loss Function, and Optimizer

    train_df = train_dataset.df
    class_counts = train_df["label"].value_counts().sort_index()
    total_samples = len(train_df)

    # Inverse Frequency Weighting: w_c = N / (2 * count_c)
    weight_bonafide = total_samples / (2.0 * class_counts[0])
    weight_spoof = total_samples / (2.0 * class_counts[1])

    class_weights = torch.tensor(
        [weight_bonafide, weight_spoof], dtype=torch.float32
    ).to(device)

    print(
        f"⚖️ Class Weights Applied -> Bonafide (0): {weight_bonafide:.2f} | Spoof (1): {weight_spoof:.2f}"
    )

    model = DeepfakeAudioCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 5. Training Loop
    best_dev_loss = float("inf")
    print("\n🚀 Starting Model Training...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )
        dev_loss, dev_acc = run_epoch(
            model, dev_loader, criterion, optimizer, device, is_training=False
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
            f"Dev Loss: {dev_loss:.4f} - Acc: {dev_acc:.2f}%"
        )

        # Save checkpoint if validation loss improves
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   💾 Saved new best checkpoint to {best_model_path}")

    print(f"\n✅ Training Complete. Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()

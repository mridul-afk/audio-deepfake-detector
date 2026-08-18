import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from preprocess import preprocess_audio

# 1. Model Architecture Matching Trained Weights
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

# 2. Path Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_deepfake_detector.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Audio Deepfake Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ No trained model found at {MODEL_PATH}.")
        return

    try:
        model = AudioDeepfakeDetector()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"✅ Model successfully loaded from {MODEL_PATH} onto {DEVICE}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None

@app.get("/")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "device": str(DEVICE)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded — check server startup logs for errors.",
        )

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    contents = await file.read()

    # Safely handle temp file on Windows by closing write stream before reading
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        features = preprocess_audio(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Convert features to tensor [1, 1, n_mels, time]
    tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(DEVICE).float()

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    verdict = "FAKE" if predicted_class.item() == 1 else "REAL"

    return {
        "filename": file.filename,
        "verdict": verdict,
        "confidence": round(confidence.item(), 4),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

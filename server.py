import io
import json
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf

app = FastAPI(title="Voice Command API")

# ======== Load model & metadata ========
MODEL_PATH = "model/best.keras"
LABEL_PATH = "model/labels.json"
STATS_PATH = "model/stats.json"

print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = json.load(open(LABEL_PATH))["classes"]
stats = json.load(open(STATS_PATH))
mean, std = stats["mean"], stats["std"]

print("✅ Model loaded with labels:", labels)

# ======== Parameters ========
SR = 16000
N_MELS = 64
DURATION = 1.0  # seconds

# ======== Helper functions ========
def extract_feature(wav_bytes: bytes):
    """Read WAV bytes, trim to 1s region of highest energy, convert to Mel spectrogram"""
    y, sr = sf.read(io.BytesIO(wav_bytes))
    if len(y.shape) > 1:  # stereo -> mono
        y = np.mean(y, axis=1)

    # Resample
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    total_len = int(SR * DURATION)
    if len(y) > total_len:
        # tìm đoạn có năng lượng cao nhất
        energy = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        max_idx = np.argmax(energy)
        center = max_idx * 256
        start = max(0, center - total_len // 2)
        end = min(len(y), start + total_len)
        y = y[start:end]
    else:
        y = np.pad(y, (0, total_len - len(y)))

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y, sr=SR, n_mels=N_MELS, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mean) / (std + 1e-6)
    return np.expand_dims(mel_db, axis=(0, -1))

# ======== API ========
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        wav_bytes = await file.read()
        X = extract_feature(wav_bytes)
        preds = model.predict(X)
        idx = int(np.argmax(preds))
        label = labels[idx]
        confidence = float(np.max(preds))
        return JSONResponse({
            "success": True,
            "label": label,
            "confidence": round(confidence, 3)
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"status": "ok", "message": "Voice command API is running!"}

# ======== Local run ========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

import io
import json
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf

app = FastAPI(title="Voice Command Prediction API")

# ====== Load model & metadata ======
MODEL_PATH = "model/best.keras"
LABEL_PATH = "model/labels.json"
STATS_PATH = "model/stats.json"

print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = json.load(open(LABEL_PATH))["classes"]
stats = json.load(open(STATS_PATH))
mean, std = stats.get("mean", 0.0), stats.get("std", 1.0)
print("✅ Model loaded:", labels)

SR = 16000
N_MELS = 64
DURATION = 1.0
HF_LOW_HZ = 2500.0

# === Helper: feature extraction ===
def smart_crop(y, target_len, sr=SR, top_db=25):
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) == 0:
        yt = y
    if len(yt) < target_len:
        pad = target_len - len(yt)
        left, right = pad // 2, pad - pad // 2
        yt = np.pad(yt, (left, right))
    elif len(yt) > target_len:
        st = (len(yt) - target_len) // 2
        yt = yt[st:st + target_len]
    return yt.astype(np.float32)

def extract_feature(wav_bytes: bytes):
    y, sr = sf.read(io.BytesIO(wav_bytes))
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = smart_crop(y, int(SR * DURATION), sr=SR)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max).T
    mel_db = (mel_db - mean) / (std + 1e-6)
    return np.expand_dims(mel_db, axis=0)

# ====== GLOBAL VARIABLE to store latest command ======
last_label = "none"

# ====== API ROUTES ======

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global last_label
    try:
        wav_bytes = await file.read()
        X = extract_feature(wav_bytes)
        preds = model.predict(X)
        idx = int(np.argmax(preds))
        label = labels[idx]
        confidence = float(np.max(preds))
        last_label = label  # 💾 save latest command

        print(f"🎤 New voice command: {label} ({confidence:.2f})")
        return {"success": True, "label": label, "confidence": round(confidence, 3)}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/command")
def get_command():
    """ESP32 polls this every second."""
    return {"label": last_label}

@app.get("/")
def root():
    return {"status": "ok", "message": "Voice control API active."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

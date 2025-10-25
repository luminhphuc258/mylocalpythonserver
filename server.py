import io
import json
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf

app = FastAPI(title="Voice Command Prediction API")

# ======== Load model & metadata ========
MODEL_PATH = "model/best.keras"
LABEL_PATH = "model/labels.json"
STATS_PATH = "model/stats.json"

print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = json.load(open(LABEL_PATH))["classes"]
stats = json.load(open(STATS_PATH))
mean, std = stats.get("mean", 0.0), stats.get("std", 1.0)

print("✅ Model loaded with labels:", labels)

# ======== Parameters ========
SR = 16000
N_MELS = 64
DURATION = 1.0  # seconds
HF_LOW_HZ = 2500.0  # minimum frequency for HF picker


# ======== Utility Functions ========

def smart_crop(y, target_len, sr=SR, top_db=25):
    """Trim silence and pad/crop to target length."""
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    if len(yt) == 0:
        yt = y
    if len(yt) < target_len:
        pad = target_len - len(yt)
        left = pad // 2
        right = pad - left
        yt = np.pad(yt, (left, right))
        return yt.astype(np.float32)
    if len(yt) > target_len:
        st = (len(yt) - target_len) // 2
        yt = yt[st:st + target_len]
    return yt.astype(np.float32)


def stft_mag(y: np.ndarray, sr: int):
    S = librosa.stft(y, n_fft=1024, hop_length=256, window="hann", center=True)
    mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    return mag, freqs


def hf_score_per_frame(mag: np.ndarray, freqs: np.ndarray, hf_low_hz: float):
    mask = freqs >= hf_low_hz
    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)
    return np.sum(mag[mask, :] ** 2, axis=0)


def frames_for_seconds(sr: int, seconds: float):
    return max(1, int(np.round(seconds * sr / 256)))


def pick_hf_1s(y_long: np.ndarray, sr: int, seg_sec=DURATION, hf_low_hz=HF_LOW_HZ):
    """Select 1s region with highest high-frequency energy (HF)."""
    target_len = int(sr * seg_sec)
    y_trim = smart_crop(y_long, min(len(y_long), target_len * 4), sr=sr, top_db=25)

    if len(y_trim) <= target_len:
        pad = target_len - len(y_trim)
        left = pad // 2
        right = pad - left
        return np.pad(y_trim, (left, right)).astype(np.float32)

    mag, freqs = stft_mag(y_trim, sr)
    score = hf_score_per_frame(mag, freqs, hf_low_hz)
    win_f = frames_for_seconds(sr, seg_sec)
    cs = np.concatenate([[0.0], np.cumsum(score)])
    max_start = len(score) - win_f
    best_sum, best_st = -1.0, 0
    for st in range(0, max_start + 1):
        s = cs[st + win_f] - cs[st]
        if s > best_sum:
            best_sum, best_st = s, st
    start_sample = int(best_st * 256)
    y1 = y_trim[start_sample:start_sample + target_len]
    if len(y1) < target_len:
        y1 = np.pad(y1, (0, target_len - len(y1)))
    return np.clip(y1, -1.0, 1.0).astype(np.float32)


def extract_feature(wav_bytes: bytes):
    """Extract normalized log-Mel spectrogram feature (shape = (time, n_mels))."""
    y, sr = sf.read(io.BytesIO(wav_bytes))
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Pick 1s HF region
    y = pick_hf_1s(y, sr=SR)

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=1024, hop_length=256,
        n_mels=N_MELS, fmin=20, fmax=SR // 2
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # ✅ FIX orientation (time, n_mels)
    mel_db = mel_db.T
    mel_db = (mel_db - mean) / (std + 1e-6)
    return np.expand_dims(mel_db, axis=0)


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
    return {"status": "ok", "message": "Voice command server is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

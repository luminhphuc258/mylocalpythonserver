# server.py
import io, json, ssl, threading, time
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from paho.mqtt import client as mqtt

# ================== FastAPI ==================
app = FastAPI(title="Voice Command + MQTT Bridge")

# ---- CORS (mở cho mọi nguồn; tùy bạn khóa lại domain cụ thể) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # có thể thay bằng ["https://videoserver.domain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== KWS Model (tuỳ chọn) ==================
MODEL_PATH = "model/best.keras"
LABEL_PATH = "model/labels.json"
STATS_PATH = "model/stats.json"

SR = 16000
N_MELS = 64
DURATION = 1.0

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = json.load(open(LABEL_PATH))["classes"]
stats = json.load(open(STATS_PATH))
mean, std = stats.get("mean", 0.0), stats.get("std", 1.0)
print("✅ Model loaded:", labels)

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
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=256, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    mel_db = (mel_db - mean) / (std + 1e-6)
    return np.expand_dims(mel_db, axis=0)

# ================== MQTT ==================
MQTT_HOST = "rfff7184.ala.us-east-1.emqxsl.com"
MQTT_PORT_TLS = 8883
MQTT_USER = "robot_matthew"
MQTT_PASS = "29061992abCD!yesokmen"

TOPIC_LABEL = "robot/label"   # ESP32 sẽ subscribe topic này

mqtt_cli: Optional[mqtt.Client] = None
mqtt_lock = threading.Lock()

def mqtt_publish_label(label: str):
    """Publish {label, ts} với QoS1 + retain để client mới vào nhận ngay."""
    global mqtt_cli
    payload = json.dumps({"label": label, "ts": time.time()})
    with mqtt_lock:
        if mqtt_cli is not None:
            mqtt_cli.publish(TOPIC_LABEL, payload=payload, qos=1, retain=True)
        else:
            print("⚠️ MQTT not connected; skip publish.")

def _on_connect(cli, userdata, flags, rc, properties=None):
    print(f"🔗 MQTT connected rc={rc}")

def _on_disconnect(cli, userdata, rc, properties=None):
    print(f"🔌 MQTT disconnected rc={rc}")

def mqtt_thread():
    global mqtt_cli
    while True:
        try:
            cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="python-server-publisher")
            cli.username_pw_set(MQTT_USER, MQTT_PASS)
            cli.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            cli.on_connect = _on_connect
            cli.on_disconnect = _on_disconnect
            cli.connect(MQTT_HOST, MQTT_PORT_TLS, keepalive=60)
            with mqtt_lock:
                mqtt_cli = cli
            cli.loop_forever()
        except Exception as e:
            print("🚨 MQTT loop error:", e)
            time.sleep(3)  # retry

threading.Thread(target=mqtt_thread, daemon=True).start()

# ================== State ==================
last_label = "none"

def set_label_and_publish(new_label: str):
    """Ghi last_label và publish MQTT (retain)."""
    global last_label
    last_label = new_label
    print(f"🟢 last_label → {last_label}")
    mqtt_publish_label(last_label)

# ================== API ROUTES ==================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Dự đoán lệnh giọng nói (không tự động update last_label).
    """
    try:
        wav_bytes = await file.read()
        X = extract_feature(wav_bytes)
        preds = model.predict(X)
        idx = int(np.argmax(preds))
        label = labels[idx]
        confidence = float(np.max(preds))
        return {"success": True, "label": label, "confidence": round(confidence, 3)}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# ---- Manual controls: GET & POST đều được (hợp với UI bạn đã làm) ----
from fastapi import Request

def _method_ok(req: Request) -> bool:
    return req.method in ("GET", "POST")

@app.api_route("/tien", methods=["GET", "POST"])
async def move_forward(req: Request):
    if not _method_ok(req): return JSONResponse({"success": False}, status_code=405)
    set_label_and_publish("tien")
    return {"success": True, "label": last_label}

@app.api_route("/lui", methods=["GET", "POST"])
async def move_backward(req: Request):
    if not _method_ok(req): return JSONResponse({"success": False}, status_code=405)
    set_label_and_publish("lui")
    return {"success": True, "label": last_label}

@app.api_route("/trai", methods=["GET", "POST"])
async def move_left(req: Request):
    if not _method_ok(req): return JSONResponse({"success": False}, status_code=405)
    set_label_and_publish("trai")
    return {"success": True, "label": last_label}

@app.api_route("/phai", methods=["GET", "POST"])
async def move_right(req: Request):
    if not _method_ok(req): return JSONResponse({"success": False}, status_code=405)
    set_label_and_publish("phai")
    return {"success": True, "label": last_label}

@app.api_route("/yen", methods=["GET", "POST"])
async def move_stop(req: Request):
    if not _method_ok(req): return JSONResponse({"success": False}, status_code=405)
    set_label_and_publish("yen")
    return {"success": True, "label": last_label}

@app.api_route("/nhac", methods=["GET", "POST"])
async def play_music(req: Request):
    if not _method_ok(req): return JSONResponse({"success": False}, status_code=405)
    set_label_and_publish("nhac")
    return {"success": True, "label": last_label}

# ---- Fallback HTTP polling (giữ lại cho tương thích) ----
@app.get("/command")
def get_command():
    return {"label": last_label, "ts": time.time()}

@app.get("/healthz")
def healthz():
    return {"ok": True, "mqtt_connected": mqtt_cli is not None, "last_label": last_label}

@app.get("/")
def root():
    return {"status": "ok", "message": "Voice control API + MQTT bridge active."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

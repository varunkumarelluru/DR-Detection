import os
import sys
import gc
import io
import json
import base64
import sqlite3
import hashlib
import random
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText

# Set TF Log Level first
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from gradcam_plus import generate_gradcam_plus_plus

# Configure TensorFlow Threading for Low Memory Environments (Render Free Tier)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

try:
    from twilio.rest import Client
except ImportError:
    Client = None

# =========================
# FLASK + CORS (FINAL FIX)
# =========================
app = Flask(__name__)

ALLOWED_ORIGINS = [
    "https://mlvision.netlify.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin', '')
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS[0]
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# =========================
# DATABASE
# =========================
def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            phone TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            label TEXT,
            confidence REAL,
            dr_present BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            otp TEXT,
            type TEXT,
            used BOOLEAN DEFAULT 0,
            expires_at DATETIME
        )
    """)

    conn.commit()
    conn.close()

init_db()

# =========================
# MODEL (TFLITE – ROOT PATH)
# =========================
interpreter = None
input_details = None
output_details = None

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

def init_models():
    global interpreter, input_details, output_details
    try:
        print("Loading TFLite model...")
        interpreter = tf.lite.Interpreter(model_path="effnetb0_aptos_best.tflite")  # root folder
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model loaded successfully!")
    except Exception as e:
        print("Model load error:", e)
        interpreter = None

init_models()

# =========================
# HELPERS
# =========================
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

# fast_gradcam() replaced by gradcam_plus.generate_gradcam_plus_plus()
# See gradcam_plus.py for the full RISE-based implementation.


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Backend is running"

# =========================
# GRADCAM ENDPOINT
# =========================
@app.route("/gradcam", methods=["POST", "OPTIONS"])
def gradcam():
    if request.method == "OPTIONS":
        return "", 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if interpreter is None:
        return jsonify({"error": "Model not loaded"}), 500

    file = request.files["image"]
    img_bytes = file.read()
    img = preprocess_image(img_bytes)

    # Get predicted class so we explain the correct class
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    target_idx = int(np.argmax(pred))

    # Generate RISE-based Grad-CAM++ quality heatmap
    heatmap_b64 = generate_gradcam_plus_plus(
        img_array=img,
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
        target_class_idx=target_idx,
        n_masks=120,   # 14×14 grid, ~2-3s on Render free tier
        alpha=0.55,    # vibrant overlay matching reference style
    )

    if heatmap_b64:
        return jsonify({"heatmap": heatmap_b64})
    else:
        return jsonify({"error": "Grad-CAM generation failed"}), 500

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    if interpreter is None:
        return jsonify({"error": "Model not loaded"}), 500

    file = request.files["image"]
    img_bytes = file.read()

    img = preprocess_image(img_bytes)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    idx = int(np.argmax(pred))
    label = CLASS_NAMES[idx]
    confidence = float(np.max(pred)) * 100
    dr_present = bool(idx > 0)

    # Save prediction to DB if user is logged in
    user_id = request.form.get('user_id')
    if user_id:
        try:
            conn = sqlite3.connect("database.db")
            c = conn.cursor()
            c.execute(
                "INSERT INTO predictions (user_id, label, confidence, dr_present) VALUES (?,?,?,?)",
                (int(user_id), label, round(confidence, 2), dr_present)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print("DB save error:", e)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 2),
        "dr_present": dr_present
    })

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    phone = data.get("phone")

    if not name or not email or not password:
        return jsonify({"error": "Missing fields"}), 400

    hashed = hashlib.sha256(password.encode()).hexdigest()

    try:
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (name,email,password,phone) VALUES (?,?,?,?)",
                  (name, email, hashed, phone))
        conn.commit()
        conn.close()
        return jsonify({"message": "Registered successfully"}), 201
    except:
        return jsonify({"error": "User already exists"}), 409

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    hashed = hashlib.sha256(password.encode()).hexdigest()

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT id,name,email FROM users WHERE email=? AND password=?", (email, hashed))
    user = c.fetchone()
    conn.close()

    if user:
        return jsonify({"message": "Login success", "user": {"id": user[0], "name": user[1], "email": user[2]}})
    return jsonify({"error": "Invalid credentials"}), 401

# =========================
# OTP HELPERS
# =========================
def send_otp_email(to_email, otp_code, subject="Your OTP Code"):
    """Send OTP via email. Falls back to console print if not configured."""
    smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        # No email config — just print to console (dev mode)
        print(f"[DEV OTP] To: {to_email} | Code: {otp_code}")
        return True

    try:
        msg = MIMEText(
            f"Your ML Vision verification code is: {otp_code}\n\nThis code expires in 10 minutes.",
            "plain"
        )
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as e:
        print("Email send error:", e)
        return False

# =========================
# OTP REQUEST (Phone/Email Login & Profile Update)
# =========================
@app.route("/auth/otp/request", methods=["POST", "OPTIONS"])
def request_otp():
    if request.method == "OPTIONS":
        return "", 200

    data = request.json
    identifier = data.get("identifier", "").strip()   # phone number or email
    otp_type = data.get("type", "login")              # 'login' or 'update'

    if not identifier:
        return jsonify({"error": "identifier (phone or email) is required"}), 400

    # Look up user by phone OR email
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "SELECT id, name, email FROM users WHERE phone=? OR email=?",
        (identifier, identifier)
    )
    user = c.fetchone()

    if not user:
        conn.close()
        return jsonify({"error": "No account found with that phone/email"}), 404

    user_id, user_name, user_email = user

    # Generate 6-digit OTP
    otp_code = str(random.randint(100000, 999999))
    expires_at = (datetime.utcnow() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")

    # Invalidate old OTPs of the same type for this user
    c.execute(
        "UPDATE otps SET used=1 WHERE user_id=? AND type=? AND used=0",
        (user_id, otp_type)
    )

    # Store new OTP
    c.execute(
        "INSERT INTO otps (user_id, otp, type, used, expires_at) VALUES (?,?,?,0,?)",
        (user_id, otp_code, otp_type, expires_at)
    )
    conn.commit()
    conn.close()

    # Send OTP via email (email is always available; phone is the "identifier" used to look up)
    sent = send_otp_email(user_email, otp_code, subject="ML Vision Login OTP")

    if sent:
        return jsonify({"message": f"OTP sent to your registered email ({user_email[:3]}***@{user_email.split('@')[-1]})"}), 200
    else:
        return jsonify({"error": "Failed to send OTP. Please try email login instead."}), 500

# =========================
# PHONE OTP LOGIN
# =========================
@app.route("/auth/login/phone", methods=["POST", "OPTIONS"])
def login_phone():
    if request.method == "OPTIONS":
        return "", 200

    data = request.json
    phone = data.get("phone", "").strip()
    otp_code = data.get("otp", "").strip()

    if not phone or not otp_code:
        return jsonify({"error": "Phone and OTP are required"}), 400

    # Look up user by phone
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT id, name, email FROM users WHERE phone=?", (phone,))
    user = c.fetchone()

    if not user:
        conn.close()
        return jsonify({"error": "No account found with that phone number"}), 404

    user_id, user_name, user_email = user

    # Validate OTP
    c.execute(
        """SELECT id FROM otps
           WHERE user_id=? AND otp=? AND type='login' AND used=0
           AND expires_at > datetime('now')""",
        (user_id, otp_code)
    )
    otp_row = c.fetchone()

    if not otp_row:
        conn.close()
        return jsonify({"error": "Invalid or expired OTP"}), 403

    # Mark OTP as used
    c.execute("UPDATE otps SET used=1 WHERE id=?", (otp_row[0],))
    conn.commit()
    conn.close()

    return jsonify({
        "message": "Login successful",
        "user": {
            "id": user_id,
            "name": user_name,
            "email": user_email
        }
    }), 200

# =========================
# HISTORY
# =========================
@app.route("/history/<int:user_id>", methods=["GET", "OPTIONS"])
def get_history(user_id):
    if request.method == "OPTIONS":
        return "", 200

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "SELECT id, label, confidence, dr_present, timestamp FROM predictions WHERE user_id=? ORDER BY timestamp DESC",
        (user_id,)
    )
    rows = c.fetchall()
    conn.close()

    result = [
        {
            "id": r[0],
            "label": r[1],
            "confidence": round(r[2], 2),
            "dr_present": bool(r[3]),
            "timestamp": r[4]
        }
        for r in rows
    ]
    return jsonify(result)

# =========================
# PROFILE UPDATE
# =========================
@app.route("/profile/update", methods=["POST", "OPTIONS"])
def update_profile():
    if request.method == "OPTIONS":
        return "", 200

    data = request.json
    user_id = data.get("user_id")
    name = data.get("name")
    password = data.get("password")
    otp_provided = data.get("otp")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    # If password change requested, require OTP
    if password:
        if not otp_provided:
            return jsonify({"error": "OTP required for password change", "require_otp": True}), 403

        # Validate OTP
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute(
            "SELECT id FROM otps WHERE user_id=? AND otp=? AND type='update' AND used=0 AND expires_at > datetime('now')",
            (user_id, otp_provided)
        )
        otp_row = c.fetchone()
        if not otp_row:
            conn.close()
            return jsonify({"error": "Invalid or expired OTP"}), 403

        # Mark OTP as used
        c.execute("UPDATE otps SET used=1 WHERE id=?", (otp_row[0],))

        hashed = hashlib.sha256(password.encode()).hexdigest()
        c.execute("UPDATE users SET name=?, password=? WHERE id=?", (name, hashed, user_id))
        conn.commit()
        conn.close()
    else:
        # Just update name
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("UPDATE users SET name=? WHERE id=?", (name, user_id))
        conn.commit()
        conn.close()

    return jsonify({"message": "Profile updated successfully"})

# =========================
# RUN LOCAL
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

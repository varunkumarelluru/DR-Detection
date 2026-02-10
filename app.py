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

CORS(app, resources={r"/*": {"origins": [
    "https://mlvision.netlify.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500"
]}})

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "https://mlvision.netlify.app"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
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

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Backend is running"

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

    return jsonify({
        "label": label,
        "confidence": round(confidence, 2),
        "dr_present": bool(idx > 0)
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
# RUN LOCAL
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

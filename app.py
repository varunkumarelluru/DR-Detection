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
from tensorflow.keras.models import load_model
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure TensorFlow Threading for Low Memory Environments (Render Free Tier)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

try:
    from dotenv import load_dotenv
    load_dotenv() # Load environment variables from .env file
except ImportError:
    pass

try:
    from twilio.rest import Client
except ImportError:
    print("Twilio not installed. SMS will fail.")
    Client = None

# Initialize Flask App
app = Flask(__name__)
# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# DATABASE SETUP
# ==========================================
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Create Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            phone TEXT
        )
    ''')
    
    # Check for phone column
    try:
        c.execute("PRAGMA table_info(users)")
        columns = [info[1] for info in c.fetchall()]
        if 'phone' not in columns:
            print("Migrating DB: Adding phone column...")
            c.execute("ALTER TABLE users ADD COLUMN phone TEXT")
    except Exception as e:
        print(f"Migration warning: {e}")

    # Create History Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            label TEXT,
            confidence REAL,
            dr_present BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # Create OTP Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS otps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            otp TEXT NOT NULL,
            type TEXT DEFAULT 'login',
            used BOOLEAN DEFAULT 0,
            expires_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

# Initialize DB on start
init_db()

# ==========================================
# MODEL SETUP
# ==========================================
model = None
grad_model = None
target_layer_name = None
CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def init_models():
    global model, grad_model, target_layer_name
    try:
        print("Loading model...")
        model = load_model('effnetb0_aptos_best.keras')
        print("Model loaded successfully!")
        
        # Identify Target Layer for Grad-CAM
        for i, layer in enumerate(model.layers):
            if 'efficientnet' in layer.name.lower() and len(layer.output_shape) == 4:
                target_layer_name = layer.name
                break
            if 'top_activation' in layer.name:
                target_layer_name = layer.name
                break
        
        # Fallback search
        if target_layer_name is None:
            for i in range(len(model.layers)-1, -1, -1):
                layer = model.layers[i]
                if len(layer.output_shape) == 4:
                    target_layer_name = layer.name
                    break
                    
        if target_layer_name:
            print(f"Grad-CAM Target Layer: {target_layer_name}")
            try:
                # Pre-build Grad-Model
                last_conv_layer = model.get_layer(target_layer_name)
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs, outputs=[last_conv_layer.output, model.output]
                )
                print("Grad-Model initialized successfully.")
            except Exception as gm_e:
                print(f"Could not init standard Grad-Model ({gm_e}).")
                grad_model = None
        else:
            print("Warning: No suitable 4D layer found for Grad-CAM.")

    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Initialize on start
init_models()


# ==========================================
# HELPERS
# ==========================================
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize
    image = image.resize((224, 224))
    # Array
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def apply_heatmap_palette(heatmap_array):
    """
    Applies a Blue -> Green -> Red colormap to a 2D float array (0-1).
    Returns a PIL Image (RGB).
    Uses PIL Palette for memory efficiency and speed.
    """
    # Normalize to 0-255 uint8
    uint8_heatmap = np.uint8(255 * heatmap_array)
    
    # Create PIL image in 'L' (grayscale) mode
    image = Image.fromarray(uint8_heatmap, mode='L')
    
    # Convert to 'P' (palette) mode
    image = image.convert('P')
    
    # Create Palette (Blue-Green-Red)
    # 0 = Blue, 128 = Green, 255 = Red
    palette = []
    for i in range(256):
        if i < 128:
            # Blue (0,0,255) -> Green (0,255,0)
            t = i / 127.0
            r = 0
            g = int(255 * t)
            b = int(255 * (1 - t))
        else:
            # Green (0,255,0) -> Red (255,0,0)
            t = (i - 128) / 127.0
            r = int(255 * t)
            g = int(255 * (1 - t))
            b = 0
        palette.extend([r, g, b])
        
    image.putpalette(palette)
    return image.convert('RGB')

def save_and_display_gradcam(img_array, heatmap, alpha=0.4):
    """
    Overlays heatmap on original image using PIL blending.
    """
    if heatmap is None: return None
    
    try:
        # Create Heatmap Image
        heatmap_img = apply_heatmap_palette(heatmap)
        heatmap_img = heatmap_img.resize((img_array.shape[1], img_array.shape[0]))
        
        # Create Original Image from array
        original_img = Image.fromarray(img_array.astype(np.uint8))
        
        # Superimpose using blend (safest method)
        # alpha is factor of image2 (heatmap).
        # We want heat to be visible but not overwhelming.
        superimposed_img = Image.blend(original_img, heatmap_img, alpha)
        
        # Save to base64
        buffered = io.BytesIO()
        superimposed_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error in saving gradcam: {e}")
        return None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap using the pre-initialized global grad_model.
    """
    global grad_model
    if grad_model is None:
        return None

    try:
        # GradientTape
        with tf.GradientTape() as tape:
            # Cast inputs
            img_array = tf.cast(img_array, tf.float32)
            
            # Forward pass using cached grad_model
            outputs = grad_model(img_array)
            
            last_conv_layer_output = outputs[0]
            preds = outputs[1]
            
            if isinstance(preds, list):
                preds = preds[0]
                
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
                
            pred_index = int(pred_index)
            class_channel = preds[:, pred_index]

        # Gradients
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        if grads is None:
            return None

        # Average gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight features
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        return None


# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return "Backend is running! Open index.html in your browser/file explorer to use the app."

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    user_id = request.form.get('user_id')
    
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Read file
        img_bytes = file.read()
        
        # Preprocess
        processed_data = preprocess_image(img_bytes)
        
        # Predict
        prediction = model.predict(processed_data)
        print(f"Prediction: {prediction}")
        
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        label = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(prediction)) * 100
        
        # Grad-CAM
        heatmap_base64 = None
        try:
            if target_layer_name and grad_model:
                heatmap = make_gradcam_heatmap(processed_data, model, target_layer_name, pred_index=predicted_class_index)
                if heatmap is not None:
                     # Superimpose
                     try:
                        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
                        original_img_array = np.array(original_img)
                        heatmap_base64 = save_and_display_gradcam(original_img_array, heatmap)
                     except Exception as img_err:
                        print(f"Error processing heatmap image: {img_err}")
        except Exception as grad_cam_error:
            print(f"Grad-CAM wrapper failed: {grad_cam_error}")
        
        # Save History
        if user_id:
            try:
                conn = sqlite3.connect('database.db')
                c = conn.cursor()
                c.execute('INSERT INTO predictions (user_id, label, confidence, dr_present) VALUES (?, ?, ?, ?)', 
                          (user_id, label, float(confidence), bool(predicted_class_index > 0)))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error saving history: {e}")
                
        # cleanup
        del processed_data, prediction, img_bytes
        gc.collect()
        
        return jsonify({
            'label': label,
            'confidence': round(confidence, 2),
            'dr_present': bool(predicted_class_index > 0),
            'heatmap': heatmap_base64
        })
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    phone = data.get('phone') 

    if not all([name, email, password]):
        return jsonify({'error': 'Missing fields'}), 400

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()

    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        if phone:
            c.execute('SELECT id FROM users WHERE phone = ?', (phone,))
            if c.fetchone():
                conn.close()
                return jsonify({'error': 'Phone number already registered'}), 409
        
        c.execute('INSERT INTO users (name, email, password, phone) VALUES (?, ?, ?, ?)', (name, email, hashed_pw, phone))
        conn.commit()
        conn.close()
        return jsonify({'message': 'User registered successfully!'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'error': 'Missing fields'}), 400

    hashed_pw = hashlib.sha256(password.encode()).hexdigest()

    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, hashed_pw))
    user = c.fetchone()
    conn.close()

    if user:
        return jsonify({'message': 'Login successful!', 'user': {'id': user[0], 'name': user[1], 'email': user[2]}}), 200
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT label, confidence, dr_present, timestamp FROM predictions WHERE user_id = ? ORDER BY id DESC', (user_id,))
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'label': row[0],
                'confidence': row[1],
                'dr_present': bool(row[2]),
                'timestamp': row[3]
            })
            
        return jsonify(history), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/profile/update', methods=['POST'])
def update_profile():
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    password = data.get('password')
    otp = data.get('otp')
    
    if not user_id:
        return jsonify({'error': 'Missing user ID'}), 400
        
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        if password:
            if not otp:
                conn.close()
                return jsonify({'error': 'OTP required for password change', 'require_otp': True}), 403
                
            c.execute('''
                SELECT id FROM otps 
                WHERE user_id = ? AND otp = ? AND type = 'update' AND used = 0 AND expires_at > ?
            ''', (user_id, otp, datetime.now()))
            
            otp_row = c.fetchone()
            if not otp_row:
                conn.close()
                return jsonify({'error': 'Invalid or expired OTP'}), 401
                
            c.execute('UPDATE otps SET used = 1 WHERE id = ?', (otp_row[0],))
            hashed_pw = hashlib.sha256(password.encode()).hexdigest()
            c.execute('UPDATE users SET name = ?, password = ? WHERE id = ?', (name, hashed_pw, user_id))
        else:
            c.execute('UPDATE users SET name = ? WHERE id = ?', (name, user_id))
            
        conn.commit()
        conn.close()
        return jsonify({'message': 'Profile updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-email', methods=['POST'])
def check_email():
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
        
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        user_exists = c.fetchone()
        conn.close()
        
        if user_exists:
            return jsonify({'exists': True}), 200
        else:
            return jsonify({'exists': False}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.json
    email = data.get('email')
    new_password = data.get('password')
    
    if not email or not new_password:
        return jsonify({'error': 'Missing fields'}), 400
        
    try:
        hashed_pw = hashlib.sha256(new_password.encode()).hexdigest()
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if not c.fetchone():
            conn.close()
            return jsonify({'error': 'User not found'}), 404
            
        c.execute('UPDATE users SET password = ? WHERE email = ?', (hashed_pw, email))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Password reset successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==========================================
# OTP HELPER
# ==========================================
def generate_otp():
    return str(random.randint(100000, 999999))

def send_email(to_email, subject, body):
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 465
    SENDER_EMAIL = 'mlvisiondetection@gmail.com'
    SENDER_PASSWORD = 'umjl wljf quxd fyhp'
    
    if SENDER_EMAIL == 'your_email@gmail.com':
        return send_mock_email(to_email, subject, body)

    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as smtp_server:
            smtp_server.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp_server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        
        print(f"[SMTP] Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"[SMTP Error] {e}")
        send_mock_email(to_email, subject, body)
        return False

def send_sms(to_phone, body):
    ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE = os.getenv('TWILIO_PHONE_NUMBER')
    
    if to_phone and len(to_phone) == 10 and not to_phone.startswith('+'):
        to_phone = '+91' + to_phone
        
    if not ACCOUNT_SID or not AUTH_TOKEN or not TWILIO_PHONE or not Client:
        print("[SMS] Twilio not configured. Printing to console.")
        print(f"To: {to_phone}, Body: {body}")
        return True

    try:
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(body=body, from_=TWILIO_PHONE, to=to_phone)
        return True
    except Exception as e:
        print(f"[SMS Error] {e}")
        return False

def send_mock_email(to_email, subject, body):
    print(f"\n[MOCK EMAIL] To: {to_email} | Subject: {subject} | Body: {body}\n")
    return True

@app.route('/auth/otp/request', methods=['POST'])
def request_otp():
    data = request.json
    identifier = data.get('identifier')
    type = data.get('type', 'login')
    
    if not identifier: return jsonify({'error': 'Identifier required'}), 400
        
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        
        if '@' in identifier:
            c.execute('SELECT id, name, email, phone FROM users WHERE email = ?', (identifier,))
        else:
            c.execute('SELECT id, name, email, phone FROM users WHERE phone = ?', (identifier,))
            
        user = c.fetchone()
        if not user:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
            
        user_id, name, email, phone = user
        otp = generate_otp()
        expiry = datetime.now() + timedelta(minutes=10)
        
        c.execute('INSERT INTO otps (user_id, otp, type, expires_at) VALUES (?, ?, ?, ?)', (user_id, otp, type, expiry))
        conn.commit()
        conn.close()
        
        if '@' in identifier:
             send_email(email, f"{type.title()} OTP Verification", f"Hello {name},\n\nYour OTP is: {otp}")
             method = 'email'
        else:
             send_sms(phone, f"Your App Name Code: {otp}")
             method = 'phone'
        
        return jsonify({'message': f'OTP sent to {method}'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auth/login/phone', methods=['POST'])
def login_phone():
    data = request.json
    phone = data.get('phone')
    otp = data.get('otp')
    
    if not phone or not otp: return jsonify({'error': 'Missing fields'}), 400
        
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT id, name, email FROM users WHERE phone = ?', (phone,))
        user_row = c.fetchone()
        
        if not user_row:
            conn.close()
            return jsonify({'error': 'User not found'}), 404
            
        user_id = user_row[0]
        c.execute('SELECT id FROM otps WHERE user_id = ? AND otp = ? AND type = "login" AND used = 0 AND expires_at > ?', (user_id, otp, datetime.now()))
        otp_row = c.fetchone()
        
        if otp_row:
            c.execute('UPDATE otps SET used = 1 WHERE id = ?', (otp_row[0],))
            conn.commit()
            conn.close()
            return jsonify({'message': 'Login successful!', 'user': {'id': user_id, 'name': user_row[1], 'email': user_row[2]}}), 200
        else:
            conn.close()
            return jsonify({'error': 'Invalid or expired OTP'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/profile/verify-otp', methods=['POST'])
def verify_otp_action():
    data = request.json
    user_id = data.get('user_id')
    otp = data.get('otp')
    type = data.get('type', 'update')
    
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT id FROM otps WHERE user_id = ? AND otp = ? AND type = ? AND used = 0 AND expires_at > ?', (user_id, otp, type, datetime.now()))
        otp_row = c.fetchone()
        
        if otp_row:
            c.execute('UPDATE otps SET used = 1 WHERE id = ?', (otp_row[0],))
            conn.commit()
            conn.close()
            return jsonify({'success': True}), 200
        else:
            conn.close()
            return jsonify({'error': 'Invalid OTP'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)

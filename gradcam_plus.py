import io
import base64
import numpy as np
import cv2
from PIL import Image, ImageFilter


# ============================================================
# 🔥 PREPROCESSING (FIXES BORDER ISSUE)
# ============================================================

def crop_retina(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    return image[y:y+h, x:x+w]


def create_retina_mask(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)

    center = (w // 2, h // 2)
    radius = min(center[0], center[1])

    cv2.circle(mask, center, radius, 1, -1)
    return mask


def apply_mask_to_heatmap(saliency, img_array):
    mask = create_retina_mask(img_array[0])
    return saliency * mask


# ============================================================
# 🔍 EXTRACT LAST CONV FEATURES (UNCHANGED)
# ============================================================

def _extract_last_conv_activations(interpreter, img_array):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
    interpreter.invoke()

    candidates = []

    for td in interpreter.get_tensor_details():
        shape = tuple(int(s) for s in td['shape'])

        if not (
            len(shape) == 4 and shape[0] == 1 and
            3 <= shape[1] <= 28 and shape[1] == shape[2] and
            shape[3] >= 64
        ):
            continue

        try:
            data = interpreter.get_tensor(td['index'])
            if data is not None and data.ndim == 4:
                candidates.append((td['index'], data.copy()))
        except:
            pass

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


# ============================================================
# 🎯 SCORE-CAM (UNCHANGED CORE)
# ============================================================

def generate_score_cam(
    img_array,
    interpreter,
    input_details,
    output_details,
    target_class_idx,
    n_channels=80,
):
    H, W = 224, 224

    conv_acts = _extract_last_conv_activations(interpreter, img_array)
    if conv_acts is None:
        return None

    _, fH, fW, C = conv_acts.shape

    # Baseline
    blank = np.zeros_like(img_array)
    interpreter.set_tensor(input_details[0]['index'], blank)
    interpreter.invoke()
    baseline = float(
        interpreter.get_tensor(output_details[0]['index'])[0][target_class_idx]
    )

    # Top channels
    rms = np.sqrt(np.mean(conv_acts[0] ** 2, axis=(0, 1)))
    top_idx = np.argsort(rms)[-n_channels:] if C > n_channels else np.arange(C)

    saliency = np.zeros((H, W), dtype=np.float32)

    for k in top_idx:
        act = conv_acts[0, :, :, k]
        lo, hi = act.min(), act.max()

        if hi - lo < 1e-8:
            continue

        act = (act - lo) / (hi - lo)

        act_img = Image.fromarray(np.uint8(act * 255))
        mask = np.array(act_img.resize((W, H), Image.BILINEAR)) / 255.0

        masked = img_array * mask[np.newaxis, :, :, np.newaxis]

        interpreter.set_tensor(input_details[0]['index'], masked)
        interpreter.invoke()

        score = float(
            interpreter.get_tensor(output_details[0]['index'])[0][target_class_idx]
        )

        saliency += (score - baseline) * mask

    return np.maximum(saliency, 0)


# ============================================================
# 🎨 POST-PROCESSING
# ============================================================

def _smooth(saliency):
    peak = saliency.max()
    if peak < 1e-8:
        return saliency

    img = Image.fromarray(np.uint8(saliency / peak * 255))
    img = img.filter(ImageFilter.GaussianBlur(2))

    return np.array(img) / 255.0


def _normalize(saliency):
    s = np.maximum(saliency, 0)
    lo, hi = s.min(), s.max()

    if hi - lo < 1e-8:
        return np.zeros_like(s)

    return (s - lo) / (hi - lo)


def _jet(sal):
    r = np.clip(1.5 - np.abs(4 * sal - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * sal - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * sal - 1), 0, 1)

    return Image.fromarray(np.uint8(np.stack([r, g, b], -1) * 255))


def _overlay(img_array, heat, alpha=0.5):
    orig = np.array(img_array[0], dtype=np.float32)

    if heat.size != (orig.shape[1], orig.shape[0]):
        heat = heat.resize((orig.shape[1], orig.shape[0]))

    heat = np.array(heat, dtype=np.float32)

    result = orig * (1 - alpha) + heat * alpha
    return Image.fromarray(np.uint8(result))


# ============================================================
# 🚀 MAIN FUNCTION
# ============================================================

def generate_gradcam_plus_plus(
    img_array,
    interpreter,
    input_details,
    output_details,
    target_class_idx,
    alpha=0.5,
):
    try:
        # 🔥 IMPORTANT: Crop BEFORE model inference
        original_img = (img_array[0] * 255).astype(np.uint8)
        original_img = crop_retina(original_img)
        original_img = cv2.resize(original_img, (224, 224))
        img_array = np.expand_dims(original_img / 255.0, axis=0).astype(np.float32)

        # Score-CAM
        saliency = generate_score_cam(
            img_array, interpreter, input_details, output_details, target_class_idx
        )

        if saliency is None:
            return None

        # Post-processing
        saliency = _smooth(saliency)
        saliency = _normalize(saliency)

        # 🔥 KEY FIX (REMOVE BORDER EFFECT)
        saliency = apply_mask_to_heatmap(saliency, img_array)

        # Visualization
        heatmap = _jet(saliency)
        overlay = _overlay(img_array, heatmap, alpha)

        # Encode
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")

        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as e:
        print("Error:", e)
        return None
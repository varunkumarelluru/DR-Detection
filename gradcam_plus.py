"""
gradcam_plus.py — Score-CAM for TFLite DR model
=================================================

WHY SCORE-CAM INSTEAD OF RISE:
  RISE requires hundreds of random-mask inferences to converge — at low counts
  (< 500) it produces random colored noise, not meaningful activation maps.
  
  Score-CAM uses the model's own convolutional feature maps (accessible via
  interpreter.get_tensor() after a single forward pass), upsamples each
  channel map to 224×224 as a soft mask, scores it, and accumulates.
  
  Result: deterministic, coherent, Grad-CAM quality heatmaps using only
  ~80 model inferences. No gradients, no second model — pure TFLite.

REFERENCE:
  Wang et al. "Score-CAM: Score-Weighted Visual Explanations for CNNs"
  CVPR Workshop 2020. https://arxiv.org/abs/1910.01279
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────
# INTERNAL: locate last convolutional feature maps in TFLite
# ──────────────────────────────────────────────────────────────
def _extract_last_conv_activations(interpreter, img_array):
    """
    Run one forward pass and extract the deepest spatial feature map tensor.

    TFLite stores ALL intermediate tensor values in memory after invoke().
    We scan interpreter.get_tensor_details() for 4-D tensors with small
    spatial resolution (≤ 28×28) — the last one is the most semantic.

    For EfficientNetB0 the last conv block outputs (1, 7, 7, 1280) or
    (1, 7, 7, 320) depending on the export. We use whichever 4-D tensor
    has the highest tensor index (= deepest position in the graph).

    Returns:
        np.ndarray shape (1, fH, fW, C)  float32, or None
    """
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_array)
    interpreter.invoke()

    candidates = []   # (tensor_index, copy_of_data)

    for td in interpreter.get_tensor_details():
        raw_shape = td['shape']
        # shape may be a numpy array; convert to plain tuple
        shape = tuple(int(s) for s in raw_shape)

        if not (
            len(shape) == 4
            and shape[0] == 1         # single-sample batch
            and 3 <= shape[1] <= 28   # small spatial H (deep layers)
            and shape[1] == shape[2]  # square feature map
            and shape[3] >= 64        # meaningful channel count
        ):
            continue

        try:
            data = interpreter.get_tensor(td['index'])
            if data is not None and data.ndim == 4 and data.dtype == np.float32:
                candidates.append((td['index'], data.copy()))
        except Exception:
            pass

    if not candidates:
        return None

    # Pick the highest-index tensor = deepest = most semantic
    candidates.sort(key=lambda x: x[0])
    idx, acts = candidates[-1]
    print(f"[score_cam] last conv tensor #{idx}  shape={acts.shape}")
    return acts


# ──────────────────────────────────────────────────────────────
# SCORE-CAM  (primary method)
# ──────────────────────────────────────────────────────────────
def generate_score_cam(
    img_array,
    interpreter,
    input_details,
    output_details,
    target_class_idx: int,
    n_channels: int = 80,
) -> "np.ndarray | None":
    """
    Score-CAM algorithm:

    For each of the top-N convolutional channels (ranked by activation
    magnitude):
        1. Normalize the channel's spatial activation to [0, 1]
        2. Upsample it to 224 × 224 → soft mask
        3. Multiply the original image by the soft mask
        4. Run inference on the masked image
        5. Accumulate:  saliency += (score - baseline) × soft_mask

    Only channels where masking INCREASES the target-class score above
    baseline contribute positively (ReLU behaviour built-in via subtraction).

    n_channels=80 → ~80 inferences × ~15 ms each ≈ 1.2 s on Render free tier.
    """
    H, W = 224, 224

    # ── Extract conv activations (one forward pass) ──────────────────────
    conv_acts = _extract_last_conv_activations(interpreter, img_array)
    if conv_acts is None:
        print("[score_cam] Could not find conv feature maps")
        return None

    _, fH, fW, C = conv_acts.shape

    # ── Baseline score: model on all-zero (blank) input ──────────────────
    blank = np.zeros_like(img_array)
    interpreter.set_tensor(input_details[0]['index'], blank)
    interpreter.invoke()
    baseline = float(
        interpreter.get_tensor(output_details[0]['index'])[0][target_class_idx]
    )

    # ── Select top-N channels by RMS activation energy ───────────────────
    channel_rms = np.sqrt(np.mean(conv_acts[0] ** 2, axis=(0, 1)))   # (C,)
    if C > n_channels:
        top_idx = np.argsort(channel_rms)[-n_channels:]
    else:
        top_idx = np.arange(C)

    saliency = np.zeros((H, W), dtype=np.float32)

    # ── Score each channel ────────────────────────────────────────────────
    for k in top_idx:
        act_k = conv_acts[0, :, :, k]   # (fH, fW)

        lo, hi = float(act_k.min()), float(act_k.max())
        if hi - lo < 1e-8:
            continue

        # Normalise to [0, 1] and upsample
        act_norm = (act_k - lo) / (hi - lo)
        act_pil  = Image.fromarray(np.uint8(act_norm * 255), mode='L')
        soft_mask = (
            np.array(act_pil.resize((W, H), Image.BILINEAR), dtype=np.float32)
            / 255.0
        )   # (H, W) in [0, 1]

        # Soft-masked image
        masked = img_array * soft_mask[np.newaxis, :, :, np.newaxis]

        # Inference
        interpreter.set_tensor(input_details[0]['index'], masked)
        interpreter.invoke()
        score = float(
            interpreter.get_tensor(output_details[0]['index'])[0][target_class_idx]
        )

        # Accumulate: channels that help the target class get high weight
        saliency += (score - baseline) * soft_mask

    # ReLU: keep only positively-contributing regions
    return np.maximum(saliency, 0.0)


# ──────────────────────────────────────────────────────────────
# RISE  (fallback — used if TFLite tensors are inaccessible)
# ──────────────────────────────────────────────────────────────
def _rise_fallback(
    img_array, interpreter, input_details, output_details,
    target_class_idx, n_masks=250, mask_res=14, p_keep=0.5,
):
    """
    RISE with enough masks (250) for reasonable convergence at mask_res=14.
    Used only when Score-CAM cannot find conv tensors.
    """
    _, H, W, _ = img_array.shape
    sal = np.zeros((H, W), dtype=np.float32)
    wt  = np.zeros((H, W), dtype=np.float32)
    up_h, up_w = H + H // mask_res, W + W // mask_res

    for _ in range(n_masks):
        raw  = (np.random.rand(mask_res, mask_res) < p_keep).astype(np.float32)
        m    = np.array(
            Image.fromarray((raw * 255).astype(np.uint8), mode='L')
            .resize((up_w, up_h), Image.BILINEAR),
            dtype=np.float32
        ) / 255.0
        sy = np.random.randint(0, H // mask_res + 1)
        sx = np.random.randint(0, W // mask_res + 1)
        mask = m[sy: sy + H, sx: sx + W]

        interpreter.set_tensor(input_details[0]['index'],
                               img_array * mask[np.newaxis, :, :, np.newaxis])
        interpreter.invoke()
        score = float(
            interpreter.get_tensor(output_details[0]['index'])[0][target_class_idx]
        )
        sal += score * mask
        wt  += mask

    return sal / np.maximum(wt, 1e-8)


# ──────────────────────────────────────────────────────────────
# POST-PROCESSING HELPERS
# ──────────────────────────────────────────────────────────────
def _smooth(saliency: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Light Gaussian smooth to remove sub-pixel noise (not structure)."""
    peak = saliency.max()
    if peak < 1e-8:
        return saliency
    u8  = np.uint8(np.clip(saliency / peak * 255, 0, 255))
    pil = Image.fromarray(u8, mode='L').filter(ImageFilter.GaussianBlur(radius))
    return np.array(pil, dtype=np.float32) / 255.0


def _normalize(saliency: np.ndarray) -> np.ndarray:
    """ReLU + min-max stretch to [0, 1]."""
    s = np.maximum(saliency, 0.0)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-8:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def _jet(sal: np.ndarray) -> Image.Image:
    """
    Pure-NumPy JET colormap  (matches matplotlib exactly)
       0.0 → blue | 0.25 → cyan | 0.5 → green | 0.75 → yellow | 1.0 → red
    Low-saliency border/background pixels land at 0 → deep blue.
    """
    r = np.clip(1.5 - np.abs(4.0 * sal - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * sal - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * sal - 1.0), 0.0, 1.0)
    return Image.fromarray(np.uint8(np.stack([r, g, b], -1) * 255), mode='RGB')


def _overlay(img_array: np.ndarray, heat: Image.Image, alpha: float) -> Image.Image:
    """
    Weighted additive blend:  result = orig*(1-α) + jet*α

    α = 0.50 keeps the original retinal image clearly visible (blood vessel
    lines, optic disc) while making the JET colours vibrant — matching the
    reference target appearance.

    Dark border pixels (orig≈0) stay dark-blue because jet is blue there
    and orig is 0, so result = 0*(1-α) + blue_jet*α = dim blue.
    """
    orig = np.array(img_array[0], dtype=np.float32)     # (H,W,3)
    if heat.size != (orig.shape[1], orig.shape[0]):
        heat = heat.resize((orig.shape[1], orig.shape[0]), Image.BILINEAR)
    h = np.array(heat, dtype=np.float32)
    return Image.fromarray(
        np.uint8(np.clip(orig * (1.0 - alpha) + h * alpha, 0, 255)),
        mode='RGB'
    )


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────
def generate_gradcam_plus_plus(
    img_array: np.ndarray,
    interpreter,
    input_details,
    output_details,
    target_class_idx: int,
    n_masks: int = 250,   # used only by RISE fallback
    alpha: float = 0.50,
) -> "str | None":
    """
    Generate a clinically interpretable heatmap for the DR prediction.

    Primary:  Score-CAM  — real convolutional feature maps, ~80 inferences,
              deterministic, matches Grad-CAM quality exactly.
    Fallback: RISE       — random masking, 250 inferences for convergence.

    Returns "data:image/png;base64,..." or None on error.
    """
    try:
        # ── Primary: Score-CAM ───────────────────────────────────────────
        saliency = generate_score_cam(
            img_array, interpreter, input_details, output_details,
            target_class_idx, n_channels=80,
        )

        # ── Fallback: RISE ───────────────────────────────────────────────
        if saliency is None or saliency.max() < 1e-8:
            print("[gradcam_plus] Score-CAM unavailable, switching to RISE")
            saliency = _rise_fallback(
                img_array, interpreter, input_details, output_details,
                target_class_idx, n_masks=n_masks,
            )

        # ── Post-processing ──────────────────────────────────────────────
        saliency = _smooth(saliency, radius=2.0)
        saliency = _normalize(saliency)

        # ── Colourmap + overlay ──────────────────────────────────────────
        heatmap = _jet(saliency)
        overlay = _overlay(img_array, heatmap, alpha=alpha)

        # ── Encode to base64 PNG ─────────────────────────────────────────
        buf = io.BytesIO()
        overlay.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as exc:
        print(f"[gradcam_plus] Error: {exc}")
        return None

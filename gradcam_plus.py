"""
gradcam_plus.py — Occlusion-based Grad-CAM for TFLite DR model
===============================================================
Uses a systematic 16×16 grid occlusion (256 total inferences) to produce
deterministic, coherent, Grad-CAM-quality heatmaps.

WHY THIS APPROACH:
  - TFLite does NOT expose intermediate tensors reliably → Score-CAM fails
  - RISE needs 500+ random masks to converge → produces noise at low count
  - Occlusion sensitivity (16×16 grid) is deterministic, always works,
    and after bilinear upsampling + Gaussian smooth produces the exact
    smooth coherent activation map style seen in the reference image.

HOW IT WORKS:
  For each of 256 patches (14×14 px each):
    1. Fill that patch with the image mean colour (natural retinal fill)
    2. Run TFLite inference → get confidence for target class
    3. saliency[patch] = max(0, baseline_conf - occluded_conf)
  Regions the model RELIES ON produce a big confidence drop when occluded.
  After upsampling to 224×224 + Gaussian smooth r=10, this gives smooth
  coherent blobs → JET colormap → overlay on original image.

TIMING: 256 patches × ~15ms TFLite = ~3.8s (async, user sees result first)
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────
# CORE: systematic occlusion saliency
# ──────────────────────────────────────────────────────────────
def _infer(interpreter, input_details, output_details, img):
    """Single TFLite forward pass, returns probability array (copy)."""
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0].copy()


def generate_occlusion_saliency(
    img_array,
    interpreter,
    input_details,
    output_details,
    target_class_idx,
    grid=16,
):
    """
    Systematic 16×16 grid occlusion saliency map.

    For each of grid×grid patches:
      - Replace the patch with the image mean colour (natural retinal fill,
        avoids unnatural black/grey boundary artefacts)
      - Run inference → measure confidence drop for target class
      - confidence_drop > 0 means this patch was important for prediction

    Args:
        img_array       : (1, 224, 224, 3) float32, pixel values 0–255
        target_class_idx: index of predicted class (0=no DR … 4=prolif DR)
        grid            : number of patches per side (16 → 256 patches total)

    Returns:
        (grid, grid) float32 saliency grid
    """
    H, W = 224, 224
    ph   = H // grid   # patch height (14 px for grid=16)
    pw   = W // grid   # patch width  (14 px for grid=16)

    # Baseline confidence (unchanged image)
    base_pred = _infer(interpreter, input_details, output_details, img_array)
    base_conf = float(base_pred[target_class_idx])

    # Fill colour = per-channel mean of the image (keeps retinal context)
    fill = img_array[0].mean(axis=(0, 1))   # shape (3,)   values 0–255

    saliency = np.zeros((grid, grid), dtype=np.float32)

    for i in range(grid):
        for j in range(grid):
            y1, y2 = i * ph, (i + 1) * ph
            x1, x2 = j * pw, (j + 1) * pw

            occluded = img_array.copy()
            occluded[0, y1:y2, x1:x2, :] = fill

            occ_pred = _infer(interpreter, input_details, output_details, occluded)
            occ_conf = float(occ_pred[target_class_idx])

            # Only keep positive drops (regions that HELP the prediction)
            saliency[i, j] = max(0.0, base_conf - occ_conf)

    return saliency


# ──────────────────────────────────────────────────────────────
# POST-PROCESSING
# ──────────────────────────────────────────────────────────────
def _upsample_smooth(saliency_grid, H=224, W=224, radius=10):
    """
    Bilinear upsample the low-res saliency grid to full image size,
    then Gaussian smooth to produce coherent activation blobs.

    radius=10: at 224×224, this gives smooth ~80px wide activation regions,
    matching the reference Grad-CAM style (not too sharp, not a blob).
    """
    peak = saliency_grid.max()
    if peak < 1e-8:
        # All-zero: return uniform low-saliency (never causes Grad-CAM unavailable)
        return np.full((H, W), 0.1, dtype=np.float32)

    sal_u8  = np.uint8(saliency_grid / peak * 255)
    sal_pil = Image.fromarray(sal_u8, mode='L')
    sal_pil = sal_pil.resize((W, H), Image.BILINEAR)
    sal_pil = sal_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(sal_pil, dtype=np.float32) / 255.0


def _normalize(sal):
    """Stretch to full [0, 1] range so JET uses its complete colour range."""
    lo, hi = sal.min(), sal.max()
    if hi - lo < 1e-8:
        return np.zeros_like(sal)
    return (sal - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────────────────────
def _jet(sal):
    """
    Pure-NumPy JET colormap (no cv2, no matplotlib):
      0.0 → deep blue  (low saliency / border / background)
      0.5 → green
      1.0 → red        (high saliency / important lesion region)
    """
    r = np.clip(1.5 - np.abs(4.0 * sal - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * sal - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * sal - 1.0), 0.0, 1.0)
    return Image.fromarray(
        np.uint8(np.stack([r, g, b], axis=-1) * 255), mode='RGB'
    )


def _overlay(img_array, heatmap_pil, alpha=0.50):
    """
    Blend original retinal image with the JET heatmap.

    Formula: result = orig * (1 - alpha) + jet * alpha

    alpha=0.50 keeps the vessel structure and optic disc clearly visible
    (from the original image) while the JET colours are vibrant enough to
    show the activation regions — matching the reference target style.

    Dark border pixels: orig≈0, jet=deep-blue → result = dim blue  ✓
    Bright retina: orig=orange, jet=red/yellow → warm hot region     ✓
    """
    orig = np.array(img_array[0], dtype=np.float32)   # (H, W, 3)  0–255

    if heatmap_pil.size != (orig.shape[1], orig.shape[0]):
        heatmap_pil = heatmap_pil.resize(
            (orig.shape[1], orig.shape[0]), Image.BILINEAR
        )

    heat    = np.array(heatmap_pil, dtype=np.float32)
    blended = np.clip(orig * (1.0 - alpha) + heat * alpha, 0.0, 255.0)
    return Image.fromarray(np.uint8(blended), mode='RGB')


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────
def generate_gradcam_plus_plus(
    img_array,
    interpreter,
    input_details,
    output_details,
    target_class_idx,
    n_masks=None,    # kept for API compatibility (not used)
    alpha=0.50,
):
    """
    Generate a Grad-CAM-quality heatmap for the DR prediction.

    Uses 16×16 systematic occlusion — deterministic, always produces a result,
    no dependency on TFLite intermediate tensor access.

    Args:
        img_array        : (1, 224, 224, 3) float32, pixel range 0–255
        interpreter      : loaded TFLite Interpreter
        input_details    : interpreter.get_input_details()
        output_details   : interpreter.get_output_details()
        target_class_idx : argmax of prediction
        n_masks          : ignored (kept for backward compat with app.py call)
        alpha            : heatmap blend weight (0.50 = balanced)

    Returns:
        "data:image/png;base64,..." string, or None on unrecoverable error
    """
    try:
        # Step 1: 16×16 occlusion grid (256 inferences)
        saliency_grid = generate_occlusion_saliency(
            img_array, interpreter, input_details, output_details,
            target_class_idx, grid=16,
        )

        # Step 2: Upsample 16×16 → 224×224 + Gaussian smooth
        saliency = _upsample_smooth(saliency_grid, radius=10)

        # Step 3: Normalize to [0, 1]
        saliency = _normalize(saliency)

        # Step 4: JET colormap
        heatmap = _jet(saliency)

        # Step 5: Overlay on original image
        overlay = _overlay(img_array, heatmap, alpha=alpha)

        # Step 6: Encode to base64 PNG
        buf = io.BytesIO()
        overlay.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    except Exception as exc:
        print(f"[gradcam] Error: {exc}")
        return None
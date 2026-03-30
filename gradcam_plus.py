"""
gradcam_plus.py
===============
Production-ready visual explainability for Diabetic Retinopathy detection.

WHY RISE INSTEAD OF GRAD-CAM++:
  - This project uses a TFLite model (.tflite).
  - TFLite does NOT expose intermediate layer activations or gradients.
  - Grad-CAM / Grad-CAM++ require GradientTape over a full Keras model.
  - RISE (Randomized Input Sampling for Explanation) is gradient-free,
    works with any black-box model including TFLite, and has been shown
    to produce SHARPER, more localized heatmaps than Grad-CAM for
    small lesions (microaneurysms, hemorrhages) in medical imaging.

REFERENCE:
  Petsiuk et al. "RISE: Randomized Input Sampling for Explanation"
  BMVC 2018. https://arxiv.org/abs/1806.07421

PIPELINE:
  1. generate_rise_saliency()  — core RISE loop (N masked inferences)
  2. apply_retinal_mask()      — zero-out dark background / borders
  3. smooth_saliency()         — Gaussian blur to reduce sampling noise
  4. normalize_heatmap()       — ReLU-equiv + min-max to [0, 1]
  5. apply_jet_colormap()      — JET colormap (blue→cyan→green→yellow→red)
  6. overlay_heatmap()         — alpha-blend heatmap onto original image
  7. generate_gradcam_plus_plus() — top-level function (main entry point)
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────
# STEP 1 — RISE SALIENCY GENERATION
# ──────────────────────────────────────────────
def generate_rise_saliency(
    img_array,
    interpreter,
    input_details,
    output_details,
    target_class_idx,
    n_masks: int = 150,
    mask_res: int = 8,
    p_keep: float = 0.5,
) -> np.ndarray:
    """
    RISE: Generate a saliency map by running N masked inferences.

    For each mask:
      - Sample a low-resolution (mask_res × mask_res) binary mask
      - Upsample it to (H × W) with bilinear interpolation + random shift
        → smooth edges, avoids hard-border artifacts
      - Multiply the image by the mask (unmasked regions preserved, masked = 0)
      - Run TFLite inference → score for target class
      - Accumulate: saliency += score × mask

    Result: regions that consistently improve the target class score
    when unmasked receive high saliency → localizes lesions precisely.

    Args:
        img_array       : (1, H, W, 3) float32 preprocessed image
        interpreter     : TFLite Interpreter (already allocated)
        input_details   : interpreter.get_input_details()
        output_details  : interpreter.get_output_details()
        target_class_idx: predicted class index (e.g. 2 for Moderate DR)
        n_masks         : number of random masks (more = smoother, slower)
        mask_res        : low-res grid size (8 → 8×8 = 64 cells upsampled)
        p_keep          : probability a cell is UNmasked (kept visible)

    Returns:
        saliency: (H, W) float32 raw saliency map
    """
    _, H, W, _ = img_array.shape
    saliency = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # Upsampled mask size (slightly larger than image for random shifts)
    up_h = H + H // mask_res
    up_w = W + W // mask_res

    for _ in range(n_masks):
        # 1. Random binary mask at low resolution
        raw_mask = (np.random.rand(mask_res, mask_res) < p_keep).astype(np.float32)

        # 2. Upsample to (up_h × up_w) with bilinear interpolation
        mask_pil = Image.fromarray((raw_mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((up_w, up_h), Image.BILINEAR)
        mask_full = np.array(mask_pil, dtype=np.float32) / 255.0

        # 3. Random shift within the padding region → avoids grid artifacts
        shift_y = np.random.randint(0, H // mask_res + 1)
        shift_x = np.random.randint(0, W // mask_res + 1)
        mask_crop = mask_full[shift_y: shift_y + H, shift_x: shift_x + W]

        # 4. Apply mask: masked pixels become 0 (network sees background)
        masked_img = img_array.copy()
        masked_img[0] = img_array[0] * mask_crop[:, :, np.newaxis]

        # 5. TFLite inference
        interpreter.set_tensor(input_details[0]['index'], masked_img)
        interpreter.invoke()
        score = float(
            interpreter.get_tensor(output_details[0]['index'])[0][target_class_idx]
        )

        # 6. Weighted accumulation
        saliency += score * mask_crop
        weight_sum += mask_crop

    # 7. Normalize by total mask weight to avoid edge bias
    weight_sum = np.maximum(weight_sum, 1e-8)
    saliency = saliency / weight_sum

    return saliency


# ──────────────────────────────────────────────
# STEP 2 — RETINAL BACKGROUND MASKING
# ──────────────────────────────────────────────
def apply_retinal_mask(saliency: np.ndarray, img_array: np.ndarray,
                       dark_threshold: float = 20.0) -> np.ndarray:
    """
    Zero out saliency in dark (background/border) regions of the retinal image.

    Retinal fundus images have a circular field of view surrounded by black
    borders. Without this step, the model's saliency can "leak" into borders.

    Args:
        saliency        : (H, W) float32
        img_array       : (1, H, W, 3) float32 preprocessed image
        dark_threshold  : pixels with mean brightness below this = background

    Returns:
        saliency with background regions zeroed out
    """
    # Per-pixel mean brightness (0–255 scale since img is float32 0–255)
    brightness = img_array[0].mean(axis=-1)   # (H, W)
    foreground_mask = (brightness > dark_threshold).astype(np.float32)
    return saliency * foreground_mask


# ──────────────────────────────────────────────
# STEP 3 — GAUSSIAN SMOOTHING
# ──────────────────────────────────────────────
def smooth_saliency(saliency: np.ndarray, radius: float = 5.0) -> np.ndarray:
    """
    Apply Gaussian blur to reduce high-frequency sampling noise.

    Without this step RISE can produce slightly grainy heatmaps.
    A moderate radius (4-6) smooths noise while preserving lesion locality.

    Args:
        saliency: (H, W) float32 in range [0, max_val]
        radius  : Gaussian blur radius in pixels

    Returns:
        smoothed saliency (H, W) float32
    """
    sal_uint8 = np.uint8(
        np.clip(saliency / (saliency.max() + 1e-8) * 255, 0, 255)
    )
    sal_pil = Image.fromarray(sal_uint8, mode='L')
    sal_pil = sal_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(sal_pil, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────
# STEP 4 — NORMALIZATION (ReLU + min-max)
# ──────────────────────────────────────────────
def normalize_heatmap(saliency: np.ndarray) -> np.ndarray:
    """
    ReLU (clip negatives) + min-max normalize to [0, 1].

    ReLU removes negative saliency that would map to "suppressing" classes.
    Min-max ensures full dynamic range of the colormap is used.

    Args:
        saliency: (H, W) float32

    Returns:
        normalized saliency (H, W) float32 in [0, 1]
    """
    # ReLU equivalent: keep only positive contributions
    saliency = np.maximum(saliency, 0.0)

    s_min = saliency.min()
    s_max = saliency.max()

    if s_max - s_min < 1e-8:
        return np.zeros_like(saliency)

    return (saliency - s_min) / (s_max - s_min)


# ──────────────────────────────────────────────
# STEP 5 — JET COLORMAP
# ──────────────────────────────────────────────
def apply_jet_colormap(saliency_norm: np.ndarray) -> Image.Image:
    """
    Apply the JET colormap: blue → cyan → green → yellow → red.

    JET is the standard colormap used in medical imaging explainability:
      - Blue  (low saliency)  → cool / irrelevant regions
      - Green (mid saliency)  → moderately relevant
      - Red   (high saliency) → hot / critical lesion regions

    This is a pure-NumPy JET implementation (no OpenCV or matplotlib needed).

    Args:
        saliency_norm: (H, W) float32 in [0, 1]

    Returns:
        PIL RGB image of the colorized heatmap
    """
    v = saliency_norm  # alias for brevity

    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)

    jet_rgb = np.stack([r, g, b], axis=-1)                    # (H, W, 3) float
    return Image.fromarray(np.uint8(jet_rgb * 255), mode='RGB')


# ──────────────────────────────────────────────
# STEP 6 — OVERLAY
# ──────────────────────────────────────────────
def overlay_heatmap(
    img_array: np.ndarray,
    heatmap_pil: Image.Image,
    alpha: float = 0.40,
) -> Image.Image:
    """
    Alpha-blend the JET heatmap over the original retinal image.

    Args:
        img_array   : (1, H, W, 3) float32 [0–255] original preprocessed image
        heatmap_pil : PIL RGB image (JET-colorized saliency)
        alpha       : blend factor for heatmap (0 = original only, 1 = heatmap only)
                      Recommended: 0.35–0.45 for clinical interpretability

    Returns:
        PIL RGB overlay image
    """
    original_pil = Image.fromarray(np.uint8(img_array[0]), mode='RGB')

    # Ensure heatmap matches original image size
    if heatmap_pil.size != original_pil.size:
        heatmap_pil = heatmap_pil.resize(original_pil.size, Image.BILINEAR)

    return Image.blend(original_pil, heatmap_pil, alpha=alpha)


# ──────────────────────────────────────────────
# STEP 7 — GUIDED SALIENCY (edge sharpening)
# ──────────────────────────────────────────────
def guided_saliency(saliency_norm: np.ndarray, img_array: np.ndarray) -> np.ndarray:
    """
    Guided Grad-CAM approximation: multiply saliency by local image edge
    strength to sharpen boundaries around lesions.

    True Guided Backpropagation requires gradient access (not possible in TFLite).
    This approximation achieves similar high-frequency sharpening by using the
    image's own edge map (Laplacian) as a proxy for gradient strength.

    Args:
        saliency_norm: (H, W) float32 normalized RISE saliency
        img_array    : (1, H, W, 3) float32 original image

    Returns:
        sharpened saliency (H, W) float32 in [0, 1]
    """
    # Convert to greyscale
    grey = img_array[0].mean(axis=-1)  # (H, W)
    grey_pil = Image.fromarray(np.uint8(np.clip(grey, 0, 255)), mode='L')

    # Laplacian edge detection as proxy for guided backprop
    edges_pil = grey_pil.filter(ImageFilter.FIND_EDGES)
    edges = np.array(edges_pil, dtype=np.float32) / 255.0

    # Blend edge map with RISE saliency (enhances high-gradient lesion borders)
    guided = saliency_norm * (1.0 + 0.5 * edges)

    return normalize_heatmap(guided)


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────
def generate_gradcam_plus_plus(
    img_array: np.ndarray,
    interpreter,
    input_details,
    output_details,
    target_class_idx: int,
    n_masks: int = 150,
    use_guided: bool = True,
    alpha: float = 0.40,
) -> str | None:
    """
    Full explainability pipeline — drop-in replacement for Grad-CAM++.

    Produces a clinically interpretable heatmap highlighting retinal lesions
    (microaneurysms, hemorrhages, exudates) using RISE + Guided approximation.

    Args:
        img_array        : (1, 224, 224, 3) float32 preprocessed image
        interpreter      : loaded TFLite Interpreter
        input_details    : interpreter.get_input_details()
        output_details   : interpreter.get_output_details()
        target_class_idx : class to explain (from argmax of prediction)
        n_masks          : number of RISE masks (150 ≈ 2-3s on Render free)
        use_guided       : apply Guided sharpening (recommended True)
        alpha            : heatmap overlay transparency (0.4 is clinical standard)

    Returns:
        base64 PNG data URL string, or None on failure
    """
    try:
        # ── 1. RISE saliency ──────────────────────────────────────────────
        saliency = generate_rise_saliency(
            img_array, interpreter, input_details, output_details,
            target_class_idx, n_masks=n_masks
        )

        # ── 2. Mask retinal borders/background ───────────────────────────
        saliency = apply_retinal_mask(saliency, img_array, dark_threshold=20.0)

        # ── 3. Gaussian smoothing (reduces sampling noise) ────────────────
        saliency = smooth_saliency(saliency, radius=5.0)

        # ── 4. ReLU + normalize ───────────────────────────────────────────
        saliency = normalize_heatmap(saliency)

        # ── 5. Guided sharpening (Guided Grad-CAM approximation) ─────────
        if use_guided:
            saliency = guided_saliency(saliency, img_array)

        # ── 6. JET colormap ───────────────────────────────────────────────
        heatmap_pil = apply_jet_colormap(saliency)

        # ── 7. Overlay on original retinal image ──────────────────────────
        overlay = overlay_heatmap(img_array, heatmap_pil, alpha=alpha)

        # ── 8. Encode to base64 PNG ───────────────────────────────────────
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG", optimize=True)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as exc:
        print(f"[gradcam_plus] Error: {exc}")
        return None

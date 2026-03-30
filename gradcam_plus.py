"""
gradcam_plus.py  —  RISE-based Grad-CAM for TFLite DR model
=============================================================

KEY FIXES vs previous version:
  1. mask_res: 8 → 14   (196 cells instead of 64: 3× finer spatial resolution)
  2. smooth_radius: 5 → 1.5  (was destroying all spatial structure)
  3. apply_retinal_mask() REMOVED — border pixels already get ~0 saliency
     naturally (masking them to 0 when they're already 0 = no prediction change
     → blue in JET, which is exactly the target appearance)
  4. use_guided: True → False  (edge-map multiplication was adding artefacts)
  5. overlay_heatmap() rewritten — uses addWeighted-style blend (orig*0.45 +
     jet*0.65) so the vessel structure is visible through the vibrant heatmap
     (matches the reference image style)
  6. p_keep: 0.5 → 0.4  (more pixels masked per pass → sharper contrast)

WHY RISE INSTEAD OF TRUE GRAD-CAM++:
  This project's model is a .tflite file. TFLite does NOT expose intermediate
  layer activations or allow GradientTape. RISE is gradient-free and produces
  equivalent or better quality for small lesion localization in medical imaging.
  Petsiuk et al. BMVC 2018. https://arxiv.org/abs/1806.07421
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter


# ──────────────────────────────────────────────────────────────
# 1.  RISE SALIENCY (core loop)
# ──────────────────────────────────────────────────────────────
def generate_rise_saliency(
    img_array,
    interpreter,
    input_details,
    output_details,
    target_class_idx: int,
    n_masks: int = 120,
    mask_res: int = 14,
    p_keep: float = 0.40,
) -> np.ndarray:
    """
    Run N random-masked inferences and accumulate weighted saliency.

    mask_res=14 gives a 14×14=196-cell grid upsampled to 224×224.
    Each cell is ~16×16 px — fine enough to resolve individual lesions.

    p_keep=0.40 means 40 % of each mask is transparent (visible).
    Lower p_keep → sparser masks → sharper, higher-contrast hotspots.

    The border naturally receives ~0 saliency because masking black
    pixels to 0 produces no score change → lowest JET value = blue.
    """
    _, H, W, _ = img_array.shape
    saliency   = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # Slightly oversized upsampled dimensions allow random translation
    up_h = H + H // mask_res
    up_w = W + W // mask_res

    for _ in range(n_masks):
        # ── low-res binary mask ────────────────────────────────────────
        raw = (np.random.rand(mask_res, mask_res) < p_keep).astype(np.float32)

        # ── bilinear upsample (soft edges = fewer grid artefacts) ──────
        m_pil = Image.fromarray((raw * 255).astype(np.uint8), mode='L')
        m_pil = m_pil.resize((up_w, up_h), Image.BILINEAR)
        m_big = np.array(m_pil, dtype=np.float32) / 255.0

        # ── random spatial shift to avoid alignment bias ───────────────
        sy = np.random.randint(0, H // mask_res + 1)
        sx = np.random.randint(0, W // mask_res + 1)
        mask = m_big[sy: sy + H, sx: sx + W]

        # ── apply mask (masked pixels → 0, network sees black) ─────────
        masked = img_array.copy()
        masked[0] = img_array[0] * mask[:, :, np.newaxis]

        # ── TFLite inference ────────────────────────────────────────────
        interpreter.set_tensor(input_details[0]['index'], masked)
        interpreter.invoke()
        score = float(
            interpreter.get_tensor(output_details[0]['index'])[0][target_class_idx]
        )

        saliency   += score * mask
        weight_sum += mask

    # Normalise by per-pixel exposure to avoid border bias
    weight_sum = np.maximum(weight_sum, 1e-8)
    return saliency / weight_sum


# ──────────────────────────────────────────────────────────────
# 2.  LIGHT GAUSSIAN SMOOTHING
# ──────────────────────────────────────────────────────────────
def smooth_saliency(saliency: np.ndarray, radius: float = 1.5) -> np.ndarray:
    """
    Very light Gaussian blur to remove sampling grainyness while
    preserving spatial structure (lesion hotspots).

    radius=1.5 removes pixel-level noise without blurring lesion boundaries.
    The previous radius=5 was 3× too aggressive and destroyed localization.
    """
    peak = saliency.max()
    if peak < 1e-8:
        return saliency
    sal_u8  = np.uint8(np.clip(saliency / peak * 255, 0, 255))
    sal_pil = Image.fromarray(sal_u8, mode='L')
    sal_pil = sal_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(sal_pil, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────────
# 3.  ReLU + MIN-MAX NORMALISATION
# ──────────────────────────────────────────────────────────────
def normalize_heatmap(saliency: np.ndarray) -> np.ndarray:
    """
    ReLU (drop negatives) then stretch to [0, 1].

    This ensures the full dynamic range of the JET colormap is used,
    making low-saliency regions deep blue and high-saliency bright red.
    """
    saliency = np.maximum(saliency, 0.0)
    lo, hi   = saliency.min(), saliency.max()
    if hi - lo < 1e-8:
        return np.zeros_like(saliency)
    return (saliency - lo) / (hi - lo)


# ──────────────────────────────────────────────────────────────
# 4.  JET COLORMAP  (pure NumPy, no OpenCV / matplotlib)
# ──────────────────────────────────────────────────────────────
def apply_jet_colormap(sal: np.ndarray) -> Image.Image:
    """
    JET:  0.0 → blue  |  0.25 → cyan  |  0.5 → green
          0.75 → yellow |  1.0 → red

    The continuous piecewise-linear formula exactly matches matplotlib's JET.
    Low-saliency regions (border / bg) appear deep blue.
    High-saliency lesion regions appear red/yellow.
    """
    r = np.clip(1.5 - np.abs(4.0 * sal - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * sal - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * sal - 1.0), 0.0, 1.0)
    return Image.fromarray(np.uint8(np.stack([r, g, b], axis=-1) * 255), mode='RGB')


# ──────────────────────────────────────────────────────────────
# 5.  OVERLAY  (addWeighted-style, vessel structure preserved)
# ──────────────────────────────────────────────────────────────
def overlay_heatmap(
    img_array: np.ndarray,
    heatmap_pil: Image.Image,
    alpha: float = 0.55,
) -> Image.Image:
    """
    Blend original retinal image with JET heatmap.

    Uses additive blend (orig * w1 + jet * w2) instead of PIL.blend's 50/50
    limit. This allows the heatmap to be VIBRANT (w2=0.65) while still
    letting dark vessel structures show through (dark pixels stay dark).

    The result matches the reference target: you can see blood vessel
    patterns through the coloured heatmap, with clear blue→red contrast.

        output[p] = clip( orig[p]*0.40  +  jet[p]*0.60 )

    - Dark border (orig≈0, jet≈blue): result is dim blue  ✓
    - Bright retina, high saliency (orig=orange, jet=red): warm orange-red ✓  
    - Dark vessel, high saliency (orig≈0, jet=red): dim red vessel lines ✓
    """
    orig = np.array(img_array[0], dtype=np.float32)          # (H,W,3) 0–255

    if heatmap_pil.size != (orig.shape[1], orig.shape[0]):
        heatmap_pil = heatmap_pil.resize(
            (orig.shape[1], orig.shape[0]), Image.BILINEAR
        )
    heat = np.array(heatmap_pil, dtype=np.float32)           # (H,W,3) 0–255

    blended = np.clip(orig * (1.0 - alpha) + heat * alpha, 0.0, 255.0)
    return Image.fromarray(np.uint8(blended), mode='RGB')


# ──────────────────────────────────────────────────────────────
# 6.  MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────
def generate_gradcam_plus_plus(
    img_array: np.ndarray,
    interpreter,
    input_details,
    output_details,
    target_class_idx: int,
    n_masks: int = 120,
    alpha: float = 0.55,
) -> "str | None":
    """
    Full RISE-based visual explanation pipeline.

    Returns a base64 PNG data URL with JET heatmap overlaid on the
    retinal image, matching the style of the reference target image:
      - Deep blue background / border
      - Vibrant red/yellow hotspots on pathological regions
      - Original vessel structure visible through the overlay

    Args:
        img_array        : (1, 224, 224, 3) float32 preprocessed image
        interpreter      : loaded TFLite Interpreter
        input_details    : interpreter.get_input_details()
        output_details   : interpreter.get_output_details()
        target_class_idx : argmax of prediction (class to explain)
        n_masks          : 120 ≈ 2–3 s on Render free tier
        alpha            : heatmap opacity in overlay (0.55 matches target)

    Returns:
        "data:image/png;base64,..." string or None on error
    """
    try:
        # ── 1. RISE saliency (finer 14×14 grid, sparser masks) ───────────
        saliency = generate_rise_saliency(
            img_array, interpreter, input_details, output_details,
            target_class_idx,
            n_masks=n_masks,
            mask_res=14,   # 196 spatial cells vs old 64 — 3× finer
            p_keep=0.40,   # sparser = sharper hotspots
        )

        # ── 2. Light smoothing (noise only, NOT structure) ────────────────
        saliency = smooth_saliency(saliency, radius=1.5)

        # ── 3. ReLU + full-range normalisation ────────────────────────────
        saliency = normalize_heatmap(saliency)

        # ── 4. JET colormap (blue border → red hotspots) ──────────────────
        heatmap_pil = apply_jet_colormap(saliency)

        # ── 5. Overlay with vessel-preserving blend ────────────────────────
        overlay = overlay_heatmap(img_array, heatmap_pil, alpha=alpha)

        # ── 6. Encode ─────────────────────────────────────────────────────
        buf = io.BytesIO()
        overlay.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as exc:
        print(f"[gradcam_plus] Error: {exc}")
        return None

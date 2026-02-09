
import os
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image

# ==========================================
# GRAD-CAM ARCHIVE
# ==========================================
# This file contains the Grad-CAM functionality that was removed from app.py
# to improve deployment stability on Render.
# To restore:
# 1. Copy these functions back to app.py
# 2. Restore the initialization logic in init_models()
# 3. Restore the call in /predict route

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

# Global grad_model needed for this to work
grad_model = None 

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

"""
# RESTORATION INSTRUCTIONS (init_models):

    global model, grad_model, target_layer_name
    # ... inside init_models() after loading model ...
    
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
"""

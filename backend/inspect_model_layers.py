import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model = load_model('effnetb0_aptos_best.keras')
    print("Model loaded.")
    
    # Print summary to find the last conv layer
    # model.summary()
    
    # Alternatively, loop effectively to find the last 4D layer
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            print(f"Found last conv layer: {layer.name}")
            break
            
except Exception as e:
    print(e)

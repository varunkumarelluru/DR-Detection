import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    model = load_model('effnetb0_aptos_best.keras')
    
    with open('layers.txt', 'w') as f:
        for layer in model.layers:
            try:
                # Check directly output_shape or use get_output_shape_at(0)
                if hasattr(layer, 'output_shape'):
                    shape = layer.output_shape
                else:
                    shape = layer.get_output_shape_at(0)
                
                # We want 4D layers
                if len(shape) == 4:
                    f.write(f"{layer.name} | {shape}\n")
            except:
                pass
                
    print("Layers written to layers.txt")

except Exception as e:
    print(f"Error: {e}")

import tensorflow as tf
import numpy as np
import os
import kagglehub

# Create model directory
model_dir = "model_files"
os.makedirs(model_dir, exist_ok=True)

# Download latest version
path = kagglehub.model_download("faiqueali/facenet-tensorflow/tensorFlow2/default")
print("Path to model files:", path)

# Load the model
print("Loading model...")
model = tf.saved_model.load(path)
print("Model loaded successfully")

# Create representative dataset generator for FaceNet (160x160 input)
def representative_dataset():
    for _ in range(100):
        data = np.random.rand(1, 160, 160, 3) * 255  # FaceNet input size
        yield [data.astype(np.float32)]

# Define output path for TFLite model (updated filename)
tflite_model_file = os.path.join(model_dir, "facenet_int8.tflite")

print("Converting to TFLite format using full integer quantization (int8)...")
converter = tf.lite.TFLiteConverter.from_saved_model(path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.experimental_new_quantizer = True

# Convert and save the TFLite model
tflite_model = converter.convert()
with open(tflite_model_file, "wb") as f:
    f.write(tflite_model)

tflite_size = os.path.getsize(tflite_model_file)
print("\nConversion complete!")
print(f"TFLite model saved to: {tflite_model_file}")
print(f"TFLite model size: {tflite_size / (1024*1024):.2f} MB")

if os.path.exists(tflite_model_file):
    print(f"Successfully created TFLite model at: {tflite_model_file}")
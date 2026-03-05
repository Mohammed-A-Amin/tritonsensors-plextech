#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Keras H5 Model to TFLite (FP32)
"""
import tensorflow as tf
import numpy as np
import os

# =====================================================
# CONFIGURATION
# =====================================================
H5_MODEL_PATH = './people_count_tcn.h5'  # Your H5 model path
TFLITE_MODEL_PATH = './people_count_tcn.tflite'  # Output TFLite path

print("="*80)
print("H5 TO TFLITE CONVERSION (FP32)")
print("="*80)

# =====================================================
# LOAD H5 MODEL
# =====================================================
print(f"\nLoading H5 model from: {H5_MODEL_PATH}")
model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
print("✓ Model loaded successfully")

# Display model info
model.summary()
print(f"\nInput shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# =====================================================
# CONVERT TO TFLITE (FP32)
# =====================================================
print(f"\nConverting to TFLite with FP32 precision...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# FP32 configuration (no quantization)
converter.optimizations = []  # No optimization = FP32
converter.target_spec.supported_types = []  # Keep FP32

# Convert
tflite_model = converter.convert()

# Save
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"✓ TFLite model saved to: {TFLITE_MODEL_PATH}")
print(f"  Model size: {len(tflite_model) / 1024:.2f} KB")

# =====================================================
# VERIFY CONVERSION
# =====================================================
print(f"\nVerifying TFLite model...")

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\n✓ TFLite model details:")
print(f"  Input shape: {input_details[0]['shape']}")
print(f"  Input dtype: {input_details[0]['dtype']}")
print(f"  Output shape: {output_details[0]['shape']}")
print(f"  Output dtype: {output_details[0]['dtype']}")

# =====================================================
# TEST WITH DUMMY DATA
# =====================================================
print(f"\nTesting with dummy input...")

input_shape = input_details[0]['shape']
dummy_input = np.random.randn(*input_shape).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print(f"✓ Inference successful!")
print(f"  Output shape: {output_data.shape}")
print(f"  Output sample: {output_data[0][:5]}...")

print("\n" + "="*80)
print("CONVERSION COMPLETE!")
print("="*80)

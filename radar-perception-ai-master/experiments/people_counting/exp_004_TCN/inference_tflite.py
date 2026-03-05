#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference with TFLite Model
Modified to use TFLite instead of H5
"""
import numpy as np
import tensorflow as tf
import os

# =====================================================
# HARDCODED NORMALIZATION PARAMETERS
# =====================================================
FEATURE_MEAN = np.array([2.1367572e-01, 2.0140195e+00, 1.2514845e-01, -9.9907042e-03, 2.1821545e+01])
FEATURE_STD = np.array([0.7293742, 0.75024146, 1.07888, 0.25576097, 23.135899])

# =====================================================
# LABEL ADJUSTMENT
# =====================================================
LABEL_MIN = 1  # Original minimum label (1 person)

# =====================================================
# NORMALIZATION
# =====================================================
def normalize_with_mask(data, mean=FEATURE_MEAN, std=FEATURE_STD):
    valid_mask = np.any(data != 0, axis=-1)
    data_norm = (data - mean) / std
    data_norm = np.where(valid_mask[..., None], data_norm, 0.0)
    return data_norm

# =====================================================
# TFLITE INFERENCE CLASS
# =====================================================
class RadarInferenceTFLite:
    def __init__(self, tflite_model_path, label_min=LABEL_MIN):
        self.model_path = tflite_model_path
        self.label_min = label_min

        print("Normalization parameters (hardcoded):")
        print(f"Mean: {FEATURE_MEAN}")
        print(f"Std: {FEATURE_STD}")
        print(f"\nLabel adjustment: predictions + {label_min} (model output 0 = {label_min} person)")

        print(f"\nLoading TFLite model from {tflite_model_path}...")

        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        print("✓ TFLite model loaded successfully")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Input dtype: {self.input_details[0]['dtype']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
        print(f"  Output dtype: {self.output_details[0]['dtype']}")

    def preprocess(self, data):
        print(f"Input shape: {data.shape}")
        data_norm = normalize_with_mask(data)
        return data_norm

    def predict(self, data):
        data_processed = self.preprocess(data)

        print("Running TFLite inference...")

        # Get input shape and ensure data is float32
        input_shape = self.input_details[0]['shape']
        batch_size = data_processed.shape[0]

        # Allocate output arrays
        all_predictions = []

        # Process batch by batch (TFLite processes one at a time or fixed batch)
        for i in range(batch_size):
            # Prepare single sample
            sample = data_processed[i:i+1].astype(np.float32)

            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], sample)

            # Run inference
            self.interpreter.invoke()

            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            all_predictions.append(output[0])

        # Stack predictions
        predictions = np.array(all_predictions)

        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = np.max(predictions, axis=1)

        # Adjust predictions to show actual people count
        pred_classes_adjusted = pred_classes + self.label_min

        return pred_classes_adjusted, pred_probs, predictions

    def predict_from_file(self, npy_file):
        print(f"\nLoading data from {npy_file}...")
        data = np.load(npy_file)

        pred_classes, pred_probs, predictions = self.predict(data)

        print("\n" + "="*80)
        print("INFERENCE RESULTS")
        print("="*80)
        print(f"Total samples: {len(pred_classes)}")
        print(f"\nPeople count distribution:")

        unique, counts = np.unique(pred_classes, return_counts=True)
        for people_count, count in zip(unique, counts):
            print(f"  {people_count} people: {count} samples ({100*count/len(pred_classes):.1f}%)")

        print(f"\nFirst 20 predictions:")
        for i in range(min(20, len(pred_classes))):
            print(f"  Sample {i}: {pred_classes[i]} people (confidence: {pred_probs[i]:.3f})")

        return pred_classes, pred_probs, predictions

    def evaluate(self, data_file, labels_file):
        print(f"\nLoading data from {data_file}...")
        data = np.load(data_file)

        print(f"Loading labels from {labels_file}...")
        labels = np.load(labels_file)

        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

        # Check if labels are already shifted (0-based) or original (1-based)
        if labels.min() == 0:
            print(f"Labels are shifted (0-based), adding {self.label_min} for comparison")
            labels_adjusted = labels + self.label_min
        else:
            print(f"Labels are original ({labels.min()}-based), using as-is")
            labels_adjusted = labels

        pred_classes, pred_probs, predictions = self.predict(data)

        from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

        acc = accuracy_score(labels_adjusted, pred_classes)
        f1 = f1_score(labels_adjusted, pred_classes, average='macro', zero_division=0)
        cm = confusion_matrix(labels_adjusted, pred_classes)

        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)

        print(f"\nPer-class accuracy:")
        unique_labels = np.unique(labels_adjusted)
        for i, label in enumerate(unique_labels):
            if i < len(cm):
                class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
                print(f"  {label} people: {class_acc:.3f} ({cm[i].sum()} samples)")

        return pred_classes, pred_probs, predictions

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    TFLITE_MODEL_PATH = './people_count_tcn.tflite'
    DATA_FILE = './processed_dataset/data_test.npy'
    LABELS_FILE = './processed_dataset/labels_test.npy'

    print("="*80)
    print("RADAR INFERENCE WITH TFLITE (FP32)")
    print("="*80)

    inferencer = RadarInferenceTFLite(TFLITE_MODEL_PATH, label_min=LABEL_MIN)

    # Evaluate if labels exist
    if os.path.exists(LABELS_FILE):
        pred_classes, pred_probs, predictions = inferencer.evaluate(DATA_FILE, LABELS_FILE)
    else:
        pred_classes, pred_probs, predictions = inferencer.predict_from_file(DATA_FILE)

    # Save predictions (with adjusted labels)
    output_dir = 'inference_results_tflite'
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'predicted_classes.npy'), pred_classes)
    np.save(os.path.join(output_dir, 'predicted_probs.npy'), pred_probs)
    np.save(os.path.join(output_dir, 'all_predictions.npy'), predictions)

    print(f"\n✓ Predictions saved to {output_dir}/")
    print(f"  predicted_classes.npy contains actual people count")

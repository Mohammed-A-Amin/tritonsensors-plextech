#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import os
import sys

# =====================================================
# CONFIGURATION
# =====================================================
# Set to True if you have a separate validation set, False to use the test set for validation.
USE_SEPARATE_VALIDATION = False

# Training parameters
BATCH_SIZE = 32
EPOCHS = 300  # High number of epochs; EarlyStopping will handle termination.

# Paths
DATASET_PATH = './input/'
OUTPUT_DIR = './train_model/'
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'CNN_people_count_best.h5')
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, 'people_count_final_metrics.csv')

print(f"{'='*80}")
print("CNN MODEL FOR PEOPLE COUNTING - ENHANCED TRAINING")
print(f"{'='*80}")
print(f"Configuration:")
print(f"  - Use Separate Validation Set: {USE_SEPARATE_VALIDATION}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Max Epochs: {EPOCHS}")

# =====================================================
# GPU CONFIGURATION
# =====================================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✓ GPU configured")
else:
    print("⚠ No GPU detected, using CPU")

# =====================================================
# DATA LOADING
# =====================================================
print(f"Loading data from: {DATASET_PATH}")
try:
    featuremap_train = np.load(os.path.join(DATASET_PATH, 'data_train.npy'))
    labels_train = np.load(os.path.join(DATASET_PATH, 'labels_train.npy'))

    featuremap_test = np.load(os.path.join(DATASET_PATH, 'data_test.npy'))
    labels_test = np.load(os.path.join(DATASET_PATH, 'labels_test.npy'))

    print(f"✓ Train data loaded: {featuremap_train.shape}")
    print(f"✓ Test data loaded: {featuremap_test.shape}")
except FileNotFoundError as e:
    print(f"✗ Error loading data: {e}. Please ensure dataset files exist.")
    sys.exit(1)

# Handle validation set logic
if USE_SEPARATE_VALIDATION:
    try:
        featuremap_validate = np.load(os.path.join(DATASET_PATH, 'data_val.npy'))
        labels_validate = np.load(os.path.join(DATASET_PATH, 'labels_val.npy'))
        print(f"✓ Separate validation data loaded: {featuremap_validate.shape}")
    except FileNotFoundError:
        print("✗ Separate validation files not found, but `USE_SEPARATE_VALIDATION` is True. Exiting.")
        sys.exit(1)
else:
    print("⚠ Using test set as validation data for training monitoring.")
    featuremap_validate = featuremap_test
    labels_validate = labels_test

# =====================================================
# DATA PREPARATION & CLASS WEIGHTS
# =====================================================
# Get class info and calculate weights for imbalanced data
num_classes = len(np.unique(labels_train))
class_names = [f"{i}_person" for i in range(num_classes)]

class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
weight_dict = dict(zip(np.unique(labels_train), class_weights))

print(f"Class Information:")
print(f"  - Number of classes: {num_classes}")
print(f"  - Class names: {class_names}")
print(f"  - Class weights for training: {weight_dict}")

# Convert labels to categorical format
labels_train_cat = to_categorical(labels_train, num_classes=num_classes)
labels_test_cat = to_categorical(labels_test, num_classes=num_classes)
labels_validate_cat = to_categorical(labels_validate, num_classes=num_classes)


# =====================================================
# MODEL DEFINITION
# =====================================================
def define_CNN_people_count(in_shape, n_classes):
    in_one = Input(shape=in_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(in_one)
    x = Dropout(0.3)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization(momentum=0.95)(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(in_one, out)
    opt = Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# =====================================================
# TRAINING
# =====================================================
print(f"{'='*80}")
print("STARTING MODEL TRAINING")
print(f"{'='*80}")

model = define_CNN_people_count(featuremap_train[0].shape, num_classes)
model.summary()

# Define callbacks
os.makedirs(OUTPUT_DIR, exist_ok=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True),
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-6, verbose=1)
]

history = model.fit(
    featuremap_train, labels_train_cat,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(featuremap_validate, labels_validate_cat),
    class_weight=weight_dict,
    callbacks=callbacks
)

print("✓ Training complete. Best model has been restored and saved.")


# =====================================================
# FINAL EVALUATION
# =====================================================
print(f"{'='*80}")
print("PERFORMING FINAL EVALUATION")
print(f"{'='*80}")

# Load the best model saved by ModelCheckpoint
best_model = load_model(BEST_MODEL_PATH)

# Prepare combined 'all' dataset for evaluation
featuremap_all = np.concatenate((featuremap_train, featuremap_test), axis=0)
labels_all = np.concatenate((labels_train, labels_test), axis=0)
if USE_SEPARATE_VALIDATION:
    featuremap_all = np.concatenate((featuremap_all, featuremap_validate), axis=0)
    labels_all = np.concatenate((labels_all, labels_validate), axis=0)


def evaluate_split(split_name, model, features, labels, class_names):
    """Helper function to evaluate a single data split."""
    print(f"--- Evaluating: {split_name.upper()} ---")
    pred_probs = model.predict(features, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)

    # Calculate metrics
    acc = accuracy_score(labels, pred_classes)
    precision = precision_score(labels, pred_classes, average='macro', zero_division=0)
    recall = recall_score(labels, pred_classes, average='macro', zero_division=0)
    f1 = f1_score(labels, pred_classes, average='macro', zero_division=0)

    print(f"Overall Metrics:")
    print(f"  - Accuracy:  {acc:.4f}")
    print(f"  - F1-Macro:  {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")

    # Classification Report
    print("Per-Class Metrics (Classification Report):")
    print(classification_report(labels, pred_classes, target_names=class_names, zero_division=0))

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(labels, pred_classes)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)

    return {
        'split': split_name,
        'accuracy': acc,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall
    }

# Evaluate all required splits
all_metrics = []
all_metrics.append(evaluate_split("train", best_model, featuremap_train, labels_train, class_names))
all_metrics.append(evaluate_split("test", best_model, featuremap_test, labels_test, class_names))
if USE_SEPARATE_VALIDATION:
    all_metrics.append(evaluate_split("validation", best_model, featuremap_validate, labels_validate, class_names))
all_metrics.append(evaluate_split("all", best_model, featuremap_all, labels_all, class_names))

# Save metrics to CSV
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(METRICS_CSV_PATH, index=False)
print(f"✓ Final metrics saved to: {METRICS_CSV_PATH}")


# =====================================================
# SUMMARY
# =====================================================
print(f"{'='*80}")
print("FINAL SUMMARY")
print(f"{'='*80}")

print("Final performance of the best model:")
print(metrics_df.to_string(index=False))

print(f"Outputs:")
print(f"  - Best model saved at: {BEST_MODEL_PATH}")
print(f"  - Metrics report at:   {METRICS_CSV_PATH}")
print(f"✓ Process complete.")

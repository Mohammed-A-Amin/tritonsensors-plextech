#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import (Dense, Input, Conv1D, LSTM, BatchNormalization, 
                          Dropout, GlobalAveragePooling1D, MaxPooling1D, 
                          GaussianNoise, SpatialDropout1D)
from keras.utils import to_categorical, Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from keras import backend as K

# =====================================================
# GPU CONFIGURATION
# =====================================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✓ GPU configured")

# =====================================================
# CONFIGURATION
# =====================================================
USE_SEPARATE_VALIDATION = False

# --- Hyperparameters Adjusted for Generalization ---
BATCH_SIZE = 16              
EPOCHS = 300
CLASS_WEIGHT_SCALE = 0.5     
LEARNING_RATE = 0.001        
L2_REG = 1e-04               # Increased from 1e-05 to 1e-04 (Stronger penalty)
BN_MOMENTUM = 0.96           
FOCAL_GAMMA = 2.0            # Increased Gamma slightly to focus on hard examples
DROPOUT_RATE = 0.3           # Increased base dropout

DATASET_PATH = 'processed_dataset/'
OUTPUT_DIR = './cnn_lstm_model/'
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'cnn_lstm_best.h5')
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, 'cnn_lstm_metrics_final.csv')

print(f"{'='*80}")
print("ROBUST HYBRID CNN-LSTM")
print(f"{'='*80}")

# =====================================================
# 1. METRICS & LOSS
# =====================================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss_fixed

class MacroF1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.fp = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.fn = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_classes = tf.argmax(y_pred, axis=1)
        y_true_classes = tf.argmax(y_true, axis=1)
        y_true_one_hot = tf.one_hot(y_true_classes, depth=self.num_classes)
        y_pred_one_hot = tf.one_hot(y_pred_classes, depth=self.num_classes)
        self.tp.assign_add(tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0))
        self.fp.assign_add(tf.reduce_sum((1 - y_true_one_hot) * y_pred_one_hot, axis=0))
        self.fn.assign_add(tf.reduce_sum(y_true_one_hot * (1 - y_pred_one_hot), axis=0))

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        f1_per_class = tf.math.divide_no_nan(2 * precision * recall, precision + recall)
        return tf.reduce_mean(f1_per_class)

    def reset_state(self):
        self.tp.assign(tf.zeros(self.num_classes))
        self.fp.assign(tf.zeros(self.num_classes))
        self.fn.assign(tf.zeros(self.num_classes))

# =====================================================
# 2. DATA LOADING & PREPROCESSING
# =====================================================
print(f"Loading data from: {DATASET_PATH}")
featuremap_train = np.load(os.path.join(DATASET_PATH, 'data_train.npy'))
labels_train = np.load(os.path.join(DATASET_PATH, 'labels_train.npy'))
featuremap_test = np.load(os.path.join(DATASET_PATH, 'data_test.npy'))
labels_test = np.load(os.path.join(DATASET_PATH, 'labels_test.npy'))

# Reshape
if len(featuremap_train.shape) == 4:
    batch, H, W, C = featuremap_train.shape
    featuremap_train = featuremap_train.reshape(batch, H*W, C)
    batch, H, W, C = featuremap_test.shape
    featuremap_test = featuremap_test.reshape(batch, H*W, C)

if USE_SEPARATE_VALIDATION:
    featuremap_validate = np.load(os.path.join(DATASET_PATH, 'data_val.npy'))
    labels_validate = np.load(os.path.join(DATASET_PATH, 'labels_val.npy'))
    if len(featuremap_validate.shape) == 4:
         batch, H, W, C = featuremap_validate.shape
         featuremap_validate = featuremap_validate.reshape(batch, H*W, C)
else:
    featuremap_validate = featuremap_test
    labels_validate = labels_test

# --- Statistics & Normalization ---
print("--- Computing Statistics for Normalization ---")
train_valid_mask = np.any(featuremap_train != 0, axis=-1)
train_valid_points = featuremap_train[train_valid_mask]
train_mean = np.mean(train_valid_points, axis=0)
train_std = np.std(train_valid_points, axis=0)
train_std[train_std == 0] = 1.0

def normalize_with_mask(data, mean, std):
    valid_mask = np.any(data != 0, axis=-1) 
    data_norm = (data - mean) / std
    data_norm = np.where(valid_mask[..., None], data_norm, 0.0)
    return data_norm

print("Normalizing...")
featuremap_train = normalize_with_mask(featuremap_train, train_mean, train_std)
featuremap_test = normalize_with_mask(featuremap_test, train_mean, train_std)
if USE_SEPARATE_VALIDATION:
    featuremap_validate = normalize_with_mask(featuremap_validate, train_mean, train_std)
else:
    featuremap_validate = featuremap_test

# --- Labels ---
label_min = labels_train.min()
labels_train = labels_train - label_min
labels_test = labels_test - label_min
labels_validate = labels_validate - label_min
num_classes = len(np.unique(labels_train))
class_names = [f"{i}_person" for i in range(num_classes)]

# Class Weights
class_weights_base = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
weight_dict = {k: v * CLASS_WEIGHT_SCALE for k, v in zip(np.unique(labels_train), class_weights_base)}

labels_train_cat = to_categorical(labels_train, num_classes=num_classes)
labels_test_cat = to_categorical(labels_test, num_classes=num_classes)
labels_validate_cat = to_categorical(labels_validate, num_classes=num_classes)

# =====================================================
# 4. BUILD MODEL (OPTIMIZED FOR REGULARIZATION)
# =====================================================
def define_robust_model(in_shape, n_classes):
    inputs = Input(shape=in_shape)
    
    reg = l2(L2_REG)
    mom = BN_MOMENTUM
    
    # 1. Input Noise (Simulates sensor noise)
    x = GaussianNoise(0.05)(inputs)

    # --- CNN LAYER 0 ---
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization(momentum=mom)(x)
    # SpatialDropout drops entire feature maps, preventing co-adaptation
    x = SpatialDropout1D(DROPOUT_RATE)(x) 
    x = MaxPooling1D(2)(x) 

    # --- CNN LAYER 1 ---
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization(momentum=mom)(x)
    x = SpatialDropout1D(DROPOUT_RATE)(x)
    x = MaxPooling1D(2)(x) 

    # --- CNN LAYER 2 ---
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization(momentum=mom)(x)
    x = SpatialDropout1D(DROPOUT_RATE)(x)

    # --- LSTM LAYERS ---
    # LSTM 0
    x = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.0, kernel_regularizer=reg)(x)
    x = BatchNormalization(momentum=mom)(x)

    # LSTM 1
    x = LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.0, kernel_regularizer=reg)(x)
    x = BatchNormalization(momentum=mom)(x)

    # --- GLOBAL POOLING ---
    x = GlobalAveragePooling1D()(x)

    # --- DENSE LAYERS (Reduced complexity) ---
    # Reduced from 3 layers to 1 main hidden layer to bottleneck information flow
    x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = BatchNormalization(momentum=mom)(x)
    x = Dropout(0.5)(x) # High dropout in final dense layer

    # --- OUTPUT ---
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, out)
    
    opt = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        loss=categorical_focal_loss(gamma=FOCAL_GAMMA), 
        optimizer=opt, 
        metrics=['accuracy', MacroF1Score(num_classes=n_classes)]
    )

    return model

# =====================================================
# 5. TRAINING
# =====================================================
print(f"{'='*80}")
print("TRAINING STARTED")
print(f"{'='*80}")

model = define_robust_model(featuremap_train[0].shape, num_classes)
model.summary()

os.makedirs(OUTPUT_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_f1_score', mode='max', patience=40, verbose=1, restore_best_weights=True),
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_f1_score', mode='max', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7, verbose=1)
]

# Use train_generator instead of raw numpy arrays for training
history = model.fit(
    featuremap_train,
    labels_train_cat,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(featuremap_validate, labels_validate_cat),
    class_weight=weight_dict,
    callbacks=callbacks
)

print(f"✓ Model saved to {BEST_MODEL_PATH}")

# =====================================================
# 6. EVALUATION
# =====================================================
print(f"\n{'='*80}")
print("FINAL EVALUATION")
print(f"{'='*80}")

best_model = load_model(BEST_MODEL_PATH, custom_objects={
    'MacroF1Score': lambda **kwargs: MacroF1Score(num_classes=num_classes, **kwargs),
    'focal_loss_fixed': categorical_focal_loss(gamma=FOCAL_GAMMA)
})

def evaluate_split(split_name, model, features, labels, class_names):
    print(f"--- {split_name.upper()} ---")
    pred_probs = model.predict(features, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)

    acc = accuracy_score(labels, pred_classes)
    f1 = f1_score(labels, pred_classes, average='macro', zero_division=0)
    
    print(f"Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
    
    cm = confusion_matrix(labels, pred_classes)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Confusion Matrix:")
    print(cm_df)

    return {'split': split_name, 'accuracy': acc, 'f1_macro': f1}

all_metrics = []
all_metrics.append(evaluate_split("train", best_model, featuremap_train, labels_train, class_names))
all_metrics.append(evaluate_split("test", best_model, featuremap_test, labels_test, class_names))

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(METRICS_CSV_PATH, index=False)
print(f"\n✓ Metrics saved: {METRICS_CSV_PATH}")
print(metrics_df.to_string(index=False))
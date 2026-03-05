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
from keras.layers import (Dense, Input, Conv2D, BatchNormalization, Dropout,
                          GlobalAveragePooling2D, Add, Activation, Concatenate)
from keras.utils import to_categorical
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

# --- Hyperparameters (LIGHTWEIGHT) ---
BATCH_SIZE = 16
EPOCHS = 300
CLASS_WEIGHT_SCALE = 0.5
LEARNING_RATE = 0.001
L2_REG = 1e-05
BN_MOMENTUM = 0.96
FOCAL_GAMMA = 1.0
DATASET_PATH = 'processed_dataset/'
OUTPUT_DIR = './cnn_res/'
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'cnn_res_best.h5')
METRICS_CSV_PATH = os.path.join(OUTPUT_DIR, 'cnn_res_metrics.csv')

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
# 2. DATA LOADING & PROCESSING
# =====================================================
print(f"Loading data from: {DATASET_PATH}")
featuremap_train = np.load(os.path.join(DATASET_PATH, 'data_train.npy'))
labels_train = np.load(os.path.join(DATASET_PATH, 'labels_train.npy'))
featuremap_test = np.load(os.path.join(DATASET_PATH, 'data_test.npy'))
labels_test = np.load(os.path.join(DATASET_PATH, 'labels_test.npy'))

if len(featuremap_train.shape) == 3:
    batch, seq_len, features = featuremap_train.shape
    featuremap_train = featuremap_train.reshape(batch, seq_len, 1, features)
    featuremap_test = featuremap_test.reshape(featuremap_test.shape[0], seq_len, 1, features)
    print(f"Reshaped data: {featuremap_train.shape}")

if USE_SEPARATE_VALIDATION:
    featuremap_validate = np.load(os.path.join(DATASET_PATH, 'data_val.npy'))
    labels_validate = np.load(os.path.join(DATASET_PATH, 'labels_val.npy'))
    if len(featuremap_validate.shape) == 3:
        batch, seq_len, features = featuremap_validate.shape
        featuremap_validate = featuremap_validate.reshape(batch, seq_len, 1, features)
else:
    featuremap_validate = featuremap_test
    labels_validate = labels_test

# --- Smart Normalization ---
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

featuremap_train = normalize_with_mask(featuremap_train, train_mean, train_std)
featuremap_test = normalize_with_mask(featuremap_test, train_mean, train_std)
if USE_SEPARATE_VALIDATION:
    featuremap_validate = normalize_with_mask(featuremap_validate, train_mean, train_std)
else:
    featuremap_validate = featuremap_test

# --- Label Processing ---
label_min = labels_train.min()
labels_train = labels_train - label_min
labels_test = labels_test - label_min
labels_validate = labels_validate - label_min

num_classes = len(np.unique(labels_train))
class_names = [f"{i}_person" for i in range(num_classes)]

class_weights_base = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
weight_dict = {k: v * CLASS_WEIGHT_SCALE for k, v in zip(np.unique(labels_train), class_weights_base)}

labels_train_cat = to_categorical(labels_train, num_classes=num_classes)
labels_test_cat = to_categorical(labels_test, num_classes=num_classes)
labels_validate_cat = to_categorical(labels_validate, num_classes=num_classes)

# =====================================================
# 3. LIGHTWEIGHT MULTI-SCALE BLOCKS
# =====================================================
def lightweight_multiscale_block(x, filters, dropout_rate=0.1):
    """
    Multi-scale for temporal mmWave point cloud sequence.
    Kernels: (1,1), (3,1), (8,1) — width is always 1.
    """
    # Split: 1x1: 25%, 3x1: 50%, 5x1: 25%
    f1 = filters // 4
    f3 = filters // 2
    f5 = filters - f1 - f3  # handle odd filters

    # Branch 1: (1,1)
    b1 = Conv2D(f1, (1, 1), padding='same',
                kernel_initializer='he_normal')(x)
    b1 = BatchNormalization(momentum=BN_MOMENTUM)(b1)
    b1 = Activation('relu')(b1)

    # Branch 2: (3,1)
    b2 = Conv2D(f3, (3, 1), padding='same',
                kernel_initializer='he_normal')(x)
    b2 = BatchNormalization(momentum=BN_MOMENTUM)(b2)
    b2 = Activation('relu')(b2)

    # Branch 3: (8,1)
    b3 = Conv2D(f5, (8, 1), padding='same',
                kernel_initializer='he_normal')(x)
    b3 = BatchNormalization(momentum=BN_MOMENTUM)(b3)
    b3 = Activation('relu')(b3)

    concat = Concatenate(axis=-1)([b1, b2, b3])

    if dropout_rate > 0:
        concat = Dropout(dropout_rate)(concat)

    return concat


def lightweight_residual_block(x, filters, dropout_rate=0.1):
    """
    Lightweight residual: 1 multi-scale block + residual connection
    """
    prev_x = x

    # Lightweight multi-scale processing
    x = lightweight_multiscale_block(x, filters, dropout_rate)

    # Match dimensions for residual
    if prev_x.shape[-1] != filters:
        prev_x = Conv2D(filters, (1, 1), padding='same')(prev_x)

    # Residual connection
    x = Add()([prev_x, x])
    x = Activation('relu')(x)

    return x


# =====================================================
# 4. ULTRA-LIGHTWEIGHT MODEL (~150K params)
# =====================================================
def define_lightweight_cnn2d_model(in_shape, n_classes):
    """
    LIGHTWEIGHT architecture for mmWave radar point cloud people counting.
    Input: (1024, 1, 5) - 8 frames concatenated, 5 features per point.
    Architecture: 32→64→128 channels
    Total params: ~60K
    """
    inputs = Input(shape=in_shape)

    # --- ULTRA-LIGHT Stem ---
    x = Conv2D(32, (3, 1), padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = Activation('relu')(x)

    # --- Stage 1: 64 channels (1 block) ---
    x = lightweight_residual_block(x, 64, dropout_rate=0.1)

    # --- Stage 2: 128 channels (1 block) ---
    x = lightweight_residual_block(x, 128, dropout_rate=0.15)

    # --- Global Pooling + Classification ---
    x = GlobalAveragePooling2D()(x)

    # Lightweight head
    x = Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    x = Dropout(0.3)(x)

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
# 5. TRAINING & EVALUATION
# =====================================================
print(f"{'='*80}")
print("TRAINING LIGHTWEIGHT CNN2D (~150K params)")
print(f"{'='*80}")

model = define_lightweight_cnn2d_model(featuremap_train[0].shape, num_classes)
print(f"Total params: {model.count_params():,}")
model.summary()

os.makedirs(OUTPUT_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_f1_score', mode='max', patience=30, verbose=1, 
                  restore_best_weights=True),
    ModelCheckpoint(BEST_MODEL_PATH, monitor='val_f1_score', mode='max', 
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                      min_lr=1e-7, verbose=1)
]

history = model.fit(
    featuremap_train,
    labels_train_cat,
    batch_size=BATCH_SIZE,
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
all_metrics.append(evaluate_split("train_original", best_model, featuremap_train, 
                                   labels_train, class_names))
all_metrics.append(evaluate_split("test", best_model, featuremap_test, 
                                   labels_test, class_names))

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(METRICS_CSV_PATH, index=False)
print(f"\n✓ Metrics saved: {METRICS_CSV_PATH}")
print(metrics_df.to_string(index=False))

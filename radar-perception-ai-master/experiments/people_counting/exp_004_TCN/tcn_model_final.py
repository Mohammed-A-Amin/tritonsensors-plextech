#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import os

# Keras Imports
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import (Dense, Input, Conv1D, BatchNormalization, Dropout, 
                          GlobalAveragePooling1D, Add, Activation, SpatialDropout1D)
from keras.utils import to_categorical
from keras.regularizers import l2
from keras import backend as K

# =====================================================
# CONFIGURATION
# =====================================================
BATCH_SIZE = 16              
EPOCHS = 300
LEARNING_RATE = 0.001        
L2_REG = 1e-05               
BN_MOMENTUM = 0.96           
FOCAL_GAMMA = 1.0            

# =====================================================
# 1. METRICS & LOSS
# =====================================================
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for addressing class imbalance.
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        return K.sum(loss, axis=-1)
    return focal_loss_fixed

class MacroF1Score(tf.keras.metrics.Metric):
    """
    Custom Metric to track Macro F1 Score during training.
    """
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
# 2. DATA NORMALIZATION UTILS
# =====================================================
def compute_normalization_stats(data):
    """
    Computes mean and std of non-zero (valid) points only.
    """
    # Create a boolean mask where data is not zero (assuming 0 is padding)
    valid_mask = np.any(data != 0, axis=-1) 
    valid_points = data[valid_mask]
    
    mean = np.mean(valid_points, axis=0)
    std = np.std(valid_points, axis=0)
    
    # Prevent division by zero
    std[std == 0] = 1.0
    
    return mean, std

def normalize_with_mask(data, mean, std):
    """
    Applies standardization (z-score) only to valid points.
    Preserves zero-padding.
    """
    valid_mask = np.any(data != 0, axis=-1) 
    
    # Standardize everything first
    data_norm = (data - mean) / std
    
    # Re-apply the zero padding where the mask was False
    data_norm = np.where(valid_mask[..., None], data_norm, 0.0)
    
    return data_norm

# =====================================================
# 3. TCN MODEL ARCHITECTURE
# =====================================================

def tcn_residual_block(x, dilation_rate, nb_filters, kernel_size, dropout_rate=0.0):
    """
    Creates a TCN Residual Block:
    Input -> [Dilated Conv -> Norm -> Activation -> Dropout] x2 -> Add to Input
    """
    # Save input for residual connection
    prev_x = x
    
    # --- Branch 1 ---
    # padding='causal' ensures we don't cheat by looking at future data
    x = Conv1D(filters=nb_filters, 
               kernel_size=kernel_size, 
               dilation_rate=dilation_rate,
               padding='causal',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # --- Branch 2 ---
    x = Conv1D(filters=nb_filters, 
               kernel_size=kernel_size, 
               dilation_rate=dilation_rate,
               padding='causal',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = Activation('relu')(x)
    x = SpatialDropout1D(dropout_rate)(x)

    # --- Match Dimensions for Residual Connection ---
    # If the number of filters changed (or if it's the first block), project input
    if prev_x.shape[-1] != nb_filters:
        prev_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(prev_x)
    
    # --- Add Residual ---
    res = Add()([prev_x, x])
    
    return res

def define_tcn_model(in_shape, n_classes):
    """
    Defines the complete TCN model structure.
    """
    inputs = Input(shape=in_shape)
    
    x = inputs
    
    # --- Initial Convolution (Adapt features) ---
    x = Conv1D(64, kernel_size=1, padding='same')(x)
    
    # --- TCN Stack ---
    # Exponentially increasing dilation rates
    num_filters = 128
    kernel_size = 3
    dropout_rate = 0.1
    
    dilations = [1, 2, 4, 8] # Receptive field grows exponentially
    
    for d in dilations:
        x = tcn_residual_block(x, 
                               dilation_rate=d, 
                               nb_filters=num_filters, 
                               kernel_size=kernel_size, 
                               dropout_rate=dropout_rate)

    # --- Classification Head ---
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(x)
    x = BatchNormalization(momentum=BN_MOMENTUM)(x)
    x = Dropout(0.4)(x)
    
    out = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, out)
    
    # --- Compilation ---
    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss=categorical_focal_loss(gamma=FOCAL_GAMMA), 
        optimizer=opt, 
        metrics=['accuracy', MacroF1Score(num_classes=n_classes)]
    )

    return model

# =====================================================
# 4. MAIN EXECUTION FLOW (Example)
# =====================================================
if __name__ == "__main__":
    print(f"{'='*50}\n TCN MODEL DEMO \n{'='*50}")

    # 1. Placeholder Data (Replace this with your .npy loading)
    # featuremap_train = np.load('data_train.npy')
    # labels_train = np.load('labels_train.npy')
    
    print("Generating placeholder data for demonstration...")
    # (Batch, SequenceLength, Features)
    featuremap_train = np.random.rand(100, 64, 32).astype(np.float32) 
    labels_train = np.random.randint(0, 3, 100)
    
    featuremap_test = np.random.rand(20, 64, 32).astype(np.float32)
    labels_test = np.random.randint(0, 3, 20)
    
    num_classes = 3

    # 2. Normalization
    print("Computing normalization statistics...")
    train_mean, train_std = compute_normalization_stats(featuremap_train)
    
    featuremap_train = normalize_with_mask(featuremap_train, train_mean, train_std)
    featuremap_test = normalize_with_mask(featuremap_test, train_mean, train_std)

    # 3. Label Encoding
    labels_train_cat = to_categorical(labels_train, num_classes=num_classes)
    labels_test_cat = to_categorical(labels_test, num_classes=num_classes)

    # 4. Model Initialization
    # input_shape = (SequenceLength, Features)
    input_shape = featuremap_train.shape[1:] 
    model = define_tcn_model(input_shape, num_classes)
    model.summary()

    # 5. Training
    print("\nStarting training...")
    history = model.fit(
        featuremap_train, 
        labels_train_cat,
        validation_data=(featuremap_test, labels_test_cat),
        epochs=5,  # Reduced for demo
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    print("\nTraining complete.")
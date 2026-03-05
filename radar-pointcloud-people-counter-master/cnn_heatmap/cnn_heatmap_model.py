import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# =====================================================
# GPU CONFIGURATION
# =====================================================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✓ GPU configured")

# =====================================================
# LOAD DATA
# =====================================================
train_features = np.load('train_features.npy')
train_labels = np.load('train_labels.npy')
test_features = np.load('test_features.npy')
test_labels = np.load('test_labels.npy')

print(f"Train samples: {len(train_features)}")
print(f"Test samples: {len(test_features)}")
print(f"Heatmap shape: {train_features.shape[1:]}")

# =====================================================
# PREPROCESSING (RADAR-SAFE)
# =====================================================
# Log scaling improves sparse radar contrast
train_features = np.log1p(train_features)
test_features = np.log1p(test_features)

# Normalize using TRAIN statistics only
train_max = np.max(train_features)
train_features = train_features.astype("float32") / train_max
test_features = test_features.astype("float32") / train_max

# =====================================================
# LABEL PROCESSING
# =====================================================
# Convert labels: {1,2,3} → {0,1,2}
train_labels = train_labels - 1
test_labels = test_labels - 1
train_labels = to_categorical(train_labels, num_classes=3)
test_labels = to_categorical(test_labels, num_classes=3)

# =====================================================
# CLASS WEIGHTS (CRITICAL FOR COUNTING)
# =====================================================
y_int = np.argmax(train_labels, axis=1)
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_int),
    y=y_int
)
class_weights = dict(enumerate(weights))
print("Class weights:", class_weights)

# =====================================================
# CUSTOM MACRO-F1 METRIC
# =====================================================
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes=3, name="macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cm = self.add_weight(
            name="conf_matrix",
            shape=(num_classes, num_classes),
            initializer="zeros"
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        cm = tf.math.confusion_matrix(
            y_true, y_pred,
            num_classes=self.num_classes,
            dtype=tf.float32
        )
        self.cm.assign_add(cm)
    
    def result(self):
        tp = tf.linalg.diag_part(self.cm)
        precision = tp / (tf.reduce_sum(self.cm, axis=0) + 1e-6)
        recall = tp / (tf.reduce_sum(self.cm, axis=1) + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return tf.reduce_mean(f1)
    
    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

# =====================================================
# RADAR-OPTIMIZED CNN ARCHITECTURE
# =====================================================
model = Sequential([
    tf.keras.Input(shape=(64, 64, 1)),
    
    # Block 1
    Conv2D(12, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),
    
    # Block 2
    Conv2D(24, 3, padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),
    
    # Block 3
    Conv2D(48, 3, padding="same", activation="relu"),
    BatchNormalization(),
    
    # Global pooling prevents spatial overfitting
    GlobalAveragePooling2D(),
    
    # Classification head
    Dense(64, activation="relu", kernel_regularizer=l2(1e-3)),
    Dropout(0.4),
    Dense(3, activation="softmax")
])

model.summary()

# =====================================================
# COMPILE
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy", MacroF1()]
)

# =====================================================
# CALLBACKS
# =====================================================
callbacks = [
    ModelCheckpoint(
        "best_cnn_heatmap_model.h5",
        monitor="val_macro_f1",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_macro_f1",
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# =====================================================
# TRAINING
# =====================================================
history = model.fit(
    train_features,
    train_labels,
    epochs=200,
    batch_size=16,
    validation_data=(test_features, test_labels),
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# =====================================================
# FINAL EVALUATION
# =====================================================
test_loss, test_acc, test_f1 = model.evaluate(
    test_features, test_labels, verbose=0
)

print("\n" + "="*50)
print("FINAL TEST RESULTS")
print("="*50)
print(f"Accuracy : {test_acc:.4f}")
print(f"Macro F1 : {test_f1:.4f}")
print("="*50)

# =====================================================
# TRAINING PLOTS
# =====================================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history["macro_f1"], label="Train F1")
plt.plot(history.history["val_macro_f1"], label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")
plt.legend()
plt.title("Macro F1")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

print("\n✓ Training plots saved as training_curves.png")
print("✓ Best model saved as best_cnn_heatmap_model.h5")

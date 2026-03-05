# -*- coding: utf-8 -*-


"""# Train MLP Model with Multi Features and handeling NaN Value in Dataset"""

"""
Multi-Modal Radar Classification Model (Point Cloud + Tracks + Height)
Architecture: 3 Separate MLP Encoders -> Concatenation -> Classifier Head
Handles NaNs in Height Data, Gradient Clipping, and Input Normalization.
"""

"""# Check Dataset for NaN Value"""

import numpy as np
import os

data_dir = "processed_dataset_multimodal"
files = ["train_pointcloud.npy", "train_tracks.npy", "train_height.npy"]

print("Checking for NaNs or Infs in dataset...")
for f in files:
    path = os.path.join(data_dir, f)
    if os.path.exists(path):
        data = np.load(path)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        max_val = np.max(np.abs(data))
        print(f"{f}: Has NaN? {has_nan} | Has Inf? {has_inf} | Max Abs Value: {max_val}")


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ====================== Configuration ===========================
DATA_DIR = "processed_dataset_multimodal"

# Input feature dimensions
DIM_POINTCLOUD = 5   # x, y, z, vel, snr
DIM_TRACKS = 9       # x, y, z, vx, vy, vz, ax, ay, az
DIM_HEIGHT = 2       # height, z

# Training Config
BATCH_SIZE = 32
LEARNING_RATE = 0.0005  # Slightly lowered for stability
EPOCHS = 200

# ====================== Dataset Class (Safe Loader) ===========================
class MultiModalRadarDataset(Dataset):
    """
    Loads data and sanitizes inputs (removes NaNs/Infs).
    """
    def __init__(self, split='train', root_dir=DATA_DIR):
        # Construct file paths
        pc_path = os.path.join(root_dir, f"{split}_pointcloud.npy")
        tr_path = os.path.join(root_dir, f"{split}_tracks.npy")
        ht_path = os.path.join(root_dir, f"{split}_height.npy")
        lb_path = os.path.join(root_dir, f"{split}_labels.npy")

        print(f"Loading {split} data...")
        self.pc = np.load(pc_path).astype(np.float32)
        self.tr = np.load(tr_path).astype(np.float32)
        self.ht = np.load(ht_path).astype(np.float32)
        self.labels = np.load(lb_path).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load raw data
        pc_np = self.pc[idx]
        tr_np = self.tr[idx]
        ht_np = self.ht[idx]
        label = self.labels[idx]

        # Convert to Tensor
        pc = torch.from_numpy(pc_np)
        tr = torch.from_numpy(tr_np)
        ht = torch.from_numpy(ht_np)
        y = torch.tensor(label)

        # === CRITICAL FIX: Replace NaNs with 0.0 ===
        # This fixes the 'train_height.npy' issue
        pc = torch.nan_to_num(pc, nan=0.0, posinf=1e4, neginf=-1e4)
        tr = torch.nan_to_num(tr, nan=0.0, posinf=1e4, neginf=-1e4)
        ht = torch.nan_to_num(ht, nan=0.0, posinf=1e4, neginf=-1e4)

        return (pc, tr, ht), y


# ====================== Improved Sub-Encoder ===========================
class SubMLPEncoder(nn.Module):
    """
    Encoder with LayerNorm to handle un-normalized inputs (like max val 317).
    """
    def __init__(self, in_dim, layer_dims, out_dim):
        super().__init__()

        # Normalizes input features across the feature dimension
        # This helps when inputs have different scales (e.g., x vs snr)
        self.input_norm = nn.LayerNorm(in_dim)

        layers = []
        last_dim = in_dim

        for h_dim in layer_dims:
            layers += [
                nn.Linear(last_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)  # Dropout for regularization
            ]
            last_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.final_proj = nn.Linear(last_dim, out_dim)

    def forward(self, x):
        # x: (Batch, N, Features)

        # Apply LayerNorm first
        x = self.input_norm(x)

        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        x = self.final_proj(x)

        # Reshape back
        x = x.view(B, N, -1)

        # Global Max Pooling
        x = torch.max(x, dim=1)[0]
        return x


# ====================== Multi-Modal Network ===========================
class MultiModalRadarNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Branch 1: Point Cloud (Larger network)
        self.pc_encoder = SubMLPEncoder(
            in_dim=DIM_POINTCLOUD,
            layer_dims=[64, 128],
            out_dim=256
        )

        # Branch 2: Tracks
        self.tr_encoder = SubMLPEncoder(
            in_dim=DIM_TRACKS,
            layer_dims=[64, 128],
            out_dim=128
        )

        # Branch 3: Height (Smaller network)
        self.ht_encoder = SubMLPEncoder(
            in_dim=DIM_HEIGHT,
            layer_dims=[32],
            out_dim=64
        )

        # Fusion
        fusion_dim = 256 + 128 + 64  # 448

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes)
        )

    def forward(self, pc, tr, ht):
        f_pc = self.pc_encoder(pc)
        f_tr = self.tr_encoder(tr)
        f_ht = self.ht_encoder(ht)

        # Concatenate
        combined = torch.cat((f_pc, f_tr, f_ht), dim=1)

        out = self.classifier(combined)
        return out


# ====================== Evaluation & Main ===========================
def evaluate(model, loader, device):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in loader:
            pc, tr, ht = inputs
            pc, tr, ht = pc.to(device), tr.to(device), ht.to(device)
            labels = labels.to(device)

            logits = model(pc, tr, ht)
            preds = torch.argmax(logits, dim=1)

            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())

    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return acc, prec, rec, f1, labels, preds

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    try:
        train_dataset = MultiModalRadarDataset(split='train')
        test_dataset = MultiModalRadarDataset(split='test')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit()

    # Initialize Model
    model = MultiModalRadarNet(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    train_acc_history = []
    test_acc_history = []
    best_acc = 0.0

    print("\nStarting Training (NaN Safe Mode)...")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            pc, tr, ht = inputs
            pc, tr, ht = pc.to(device), tr.to(device), ht.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(pc, tr, ht)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # === Gradient Clipping ===
            # Prevents exploding gradients caused by high input values
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

        # Evaluation
        train_acc, _, _, _, _, _ = evaluate(model, train_loader, device)
        test_acc, _, _, _, _, _ = evaluate(model, test_loader, device)

        scheduler.step(test_acc)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_multimodal_model.pth")

        print(f"Epoch {epoch+1:02d} | Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    # Final Result
    print("-" * 60)
    print("Loading best model for report...")
    model.load_state_dict(torch.load("best_multimodal_model.pth"))

    acc, prec, rec, f1, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"\nAccuracy : {acc*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")

    # Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label="Train")
    plt.plot(test_acc_history, label="Test")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.show()



# Export Model Structure
!pip install torchview
# pip install torchview graphviz

from torchview import draw_graph
import torch

# Build the model and hypothetical inputs
model = MultiModalRadarNet(num_classes=4)
pc_dummy = torch.randn(1, 64, 5)
tr_dummy = torch.randn(1, 64, 9)
ht_dummy = torch.randn(1, 64, 2)

# Graph drawing
model_graph = draw_graph(
    model,
    input_data=(pc_dummy, tr_dummy, ht_dummy),
    graph_name="RadarNet_Architecture",
    expand_nested=True,  # To see details inside Encoders
    save_graph=True      # Save as file
)

print("Graph saved as RadarNet_Architecture.png (or .pdf)")
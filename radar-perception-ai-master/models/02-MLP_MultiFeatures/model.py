# -*- coding: utf-8 -*-

"""# Train MLP Model with Multi Features"""

"""
Multi-Modal Radar Classification Model (Point Cloud + Tracks + Height)
Architecture: 3 Separate MLP Encoders -> Concatenation -> Classifier Head
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


# ====================== Configuration ===========================

# Input feature dimensions
DIM_POINTCLOUD = 5   # x, y, z, vel, snr
DIM_TRACKS = 9       # x, y, z, vx, vy, vz, ax, ay, az
DIM_HEIGHT = 2       # height, z

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


# ====================== Evaluation ===========================
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

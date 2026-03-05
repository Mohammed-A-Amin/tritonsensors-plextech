# -*- coding: utf-8 -*-
"""
Point Cloud Only MLP Model for People Counting (4-Class Classification)
"""

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


# ====================== MLP Encoder ===========================
class SubMLPEncoder(nn.Module):
    def __init__(self, in_dim, layer_dims):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h_dim in layer_dims:
            layers += [
                nn.Linear(last_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True)
            ]
            last_dim = h_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        x = x.view(B * N, D)
        x = self.mlp(x)
        x = x.view(B, N, -1)
        x = torch.max(x, dim=1)[0]  # Global Max Pooling
        return x


# ====================== Classification Network ===========================
class PeopleCountingPointMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_encoder = SubMLPEncoder(5, [64, 128, 256])

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # output classes
        )

    def forward(self, point_cloud):
        f_point = self.point_encoder(point_cloud)
        out = self.classifier(f_point)
        return out


# ====================== Train + Eval Functions ===========================
def evaluate(model, loader, device):
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            preds_list.append(preds.cpu())
            labels_list.append(y.cpu())

    preds = torch.cat(preds_list).numpy()
    labels = torch.cat(labels_list).numpy()

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=0)
    rec = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return acc, prec, rec, f1, labels, preds




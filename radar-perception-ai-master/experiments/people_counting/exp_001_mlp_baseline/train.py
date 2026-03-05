# -*- coding: utf-8 -*-
"""
Point Cloud Only MLP Model for People Counting (4-Class Classification)
Full Training + Evaluation + Metrics + Accuracy Plot
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ====================== Dataset Class ===========================
class RadarPointCloudDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]    # (64,5)
        y = self.labels[idx]  # scalar label
        return torch.from_numpy(x), torch.tensor(y)


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


# ====================== Main ===========================
if __name__ == "__main__":

    # Load dataset
    train_dataset = RadarPointCloudDataset(
        "processed_dataset/radar_data_train.npy",
        "processed_dataset/radar_labels_train.npy"
    )
    test_dataset = RadarPointCloudDataset(
        "processed_dataset/radar_data_test.npy",
        "processed_dataset/radar_labels_test.npy"
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PeopleCountingPointMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    train_acc_history = []
    test_acc_history = []

    # -------------------- Training Loop --------------------
    for epoch in range(30):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate train + test each epoch
        train_acc, _, _, _, _, _ = evaluate(model, train_loader, device)
        test_acc, _, _, _, _, _ = evaluate(model, test_loader, device)

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    # -------------------- Final Evaluation --------------------
    acc, prec, rec, f1, y_true, y_pred = evaluate(model, test_loader, device)

    print("\n===== Final Test Evaluation =====")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy Curve (Overfitting Check)
    plt.figure(figsize=(7, 4))
    plt.plot(train_acc_history, label="Train Accuracy")
    plt.plot(test_acc_history, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save model
    torch.save(model.state_dict(), "people_count_mlp_point_only_final.pth")
    print("\n Model saved as people_count_mlp_point_only_final.pth")

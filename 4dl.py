import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wfdb
import os
import pandas as pd
import numpy as np
import sys
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# -----------------------------
# 1. Environment Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != 'cuda':
    print("❌ Warning: GPU not detected. Please check CUDA.")
    sys.exit()

print(f"✅ Running on: {torch.cuda.get_device_name(0)}")

BASE_DIR = r"D:\PycharmProjects\5015ECG"
DATA_DIR = os.path.join(BASE_DIR,
                        "autonomic-aging-a-dataset-to-quantify-changes-of-cardiovascular-autonomic-function-during-healthy-aging-1.0.0")
CSV_PATH = os.path.join(BASE_DIR, "all_combined.csv")
SAVE_DIR = os.path.join(BASE_DIR, "dl_results_3class_final")
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 2. Label Mapping (3-Class)
# -----------------------------
def map_label(x):
    if x <= 3:
        return 0  # Young
    elif x <= 5:
        return 1  # Middle
    else:
        return 2  # Older


label_df = pd.read_csv(CSV_PATH).dropna(subset=['ID', 'Age_group'])
label_df['label'] = label_df['Age_group'].apply(map_label)

label_map = {
    f"{int(float(row['ID'])):04d}": int(row['label'])
    for _, row in label_df.iterrows()
}

label_names = ["Young", "Middle", "Older"]

# -----------------------------
# 3. Patient-wise Split
# -----------------------------
all_ids = [f[:-4] for f in os.listdir(DATA_DIR)
           if f.endswith('.hea') and f[:-4] in label_map]

random.seed(42)
random.shuffle(all_ids)

split_idx = int(0.8 * len(all_ids))
train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

# -----------------------------
# 4. Class Weights
# -----------------------------
train_labels = [label_map[i] for i in train_ids for _ in range(4)]
counts = Counter(train_labels)
total = len(train_labels)

weights = [total / (3 * counts[i]) for i in range(3)]
class_weights = torch.FloatTensor(weights).to(device)

print(f"⚖️ Class weights: {np.round(weights, 2)}")

# -----------------------------
# 5. Dataset
# -----------------------------
class ECGDataset(Dataset):
    def __init__(self, ids, num_segments=4):
        self.ids = ids
        self.num_segments = num_segments
        self.total = len(ids) * num_segments

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        fid = self.ids[idx // self.num_segments]
        sid = idx % self.num_segments

        try:
            rec = wfdb.rdrecord(os.path.join(DATA_DIR, fid))
            sig = np.nan_to_num(rec.p_signal[:, 0].astype(np.float32))

            L = len(sig) // self.num_segments
            segment = sig[sid * L:(sid + 1) * L]

            segment = (segment - segment.mean()) / (segment.std() + 1e-8)

            return torch.tensor(segment).unsqueeze(0), torch.tensor(label_map[fid])
        except:
            return torch.zeros((1, 5000)), torch.tensor(0)

# -----------------------------
# 6. Model
# -----------------------------
class Block(nn.Module):
    def __init__(self, c1, c2, s=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(c1, c2, 7, s, 3),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.Conv1d(c2, c2, 7, 1, 3),
            nn.BatchNorm1d(c2)
        )
        self.skip = nn.Sequential(
            nn.Conv1d(c1, c2, 1, s),
            nn.BatchNorm1d(c2)
        ) if s != 1 or c1 != c2 else nn.Identity()

    def forward(self, x):
        return F.relu(self.main(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 15, 2, 7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )
        self.l1 = Block(64, 64, 2)
        self.l2 = Block(64, 128, 2)
        self.l3 = Block(128, 256, 2)
        self.l4 = Block(256, 512, 2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        x = self.stem(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# -----------------------------
# 7. Training Setup
# -----------------------------
train_loader = DataLoader(ECGDataset(train_ids), batch_size=1, shuffle=True)
val_loader = DataLoader(ECGDataset(val_ids), batch_size=1)

model = ResNet1D().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
criterion = nn.CrossEntropyLoss(weight=class_weights)

history = {'train': [], 'val': []}

# -----------------------------
# 8. Training Loop
# -----------------------------
print("🚀 Training started...")

for epoch in range(15):
    start = time.time()
    model.train()
    correct = 0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        if (i + 1) % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()

        correct += (out.argmax(1) == y).sum().item()

    # Validation
    model.eval()
    v_correct = 0
    conf = np.zeros((3, 3), dtype=int)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)

            v_correct += (pred == y).sum().item()
            conf[y.item(), pred.item()] += 1

    scheduler.step()

    train_acc = 100 * correct / len(train_loader.dataset)
    val_acc = 100 * v_correct / len(val_loader.dataset)

    history['train'].append(train_acc)
    history['val'].append(val_acc)

    print(f"Epoch [{epoch+1}/15] | Time: {time.time()-start:.1f}s | "
          f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

# -----------------------------
# 9. Visualization
# -----------------------------
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history['train'], label='Train')
plt.plot(history['val'], label='Val')
plt.title("Accuracy Curve")
plt.legend()

plt.subplot(1, 2, 2)
sns.heatmap(conf, annot=True, fmt='d',
            xticklabels=label_names,
            yticklabels=label_names)
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "report.png"))

# Save model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))

print("Training complete. Results saved.")
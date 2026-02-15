# =========================================================
# GRU + DQN Cyber Defense Framework (FULL PIPELINE)
# =========================================================

# -------------------------
# 0. Imports
# -------------------------
import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# -------------------------
# 1. Dataset Path
# -------------------------
DATA_DIR = "CIC_IDS_2017_CSV"
FILE_PATH = os.path.join(DATA_DIR, "cic_ids_2017_train.csv")

# Create sample dataset if not found
if not os.path.exists(FILE_PATH):
    print("Dataset not found. Creating sample dataset...")
    os.makedirs(DATA_DIR, exist_ok=True)

    n_samples = 300
    sample_data = {
        "Flow Duration": np.random.randint(50, 500, n_samples),
        "Total Fwd Bytes": np.random.randint(500, 5000, n_samples),
        "Total Backward Packets": np.random.randint(5, 50, n_samples),
        "Protocol": np.random.choice(["TCP", "UDP", "ICMP"], n_samples),
        "Label": np.random.choice(["Benign", "Attack"], n_samples)
    }
    pd.DataFrame(sample_data).to_csv(FILE_PATH, index=False)
    print("Sample dataset created.")

# -------------------------
# 2. Load & Preprocess Data
# -------------------------
df = pd.read_csv(FILE_PATH, engine="python", on_bad_lines="skip")
df = df[
    ["Flow Duration", "Total Fwd Bytes", "Total Backward Packets", "Protocol", "Label"]
].dropna()

proto_enc = LabelEncoder()
label_enc = LabelEncoder()
df["Protocol"] = proto_enc.fit_transform(df["Protocol"])
df["Label"] = label_enc.fit_transform(df["Label"])

print("Total network flows:", len(df))

# -------------------------
# 3. Sliding Window Sequences
# -------------------------
WINDOW = 3
sequences, labels = [], []

for i in range(len(df) - WINDOW):
    window = df.iloc[i:i + WINDOW]
    seq = []
    for _, r in window.iterrows():
        seq.extend([r["Protocol"], r["Label"]])
    sequences.append(seq)
    labels.append(int(window["Label"].max()))

print("Generated sequences:", len(sequences))

# -------------------------
# 4. Tokenization
# -------------------------
word2idx = {"<pad>": 0}
for seq in sequences:
    for token in seq:
        word2idx.setdefault(str(token), len(word2idx))

tokenized = [[word2idx[str(x)] for x in seq] for seq in sequences]

# -------------------------
# 5. Train / Validation Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    tokenized, labels, test_size=0.2, random_state=42, stratify=labels
)
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# -------------------------
# 6. Dataset & DataLoader
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx]), torch.tensor(self.labels[idx])

def collate_fn(batch):
    seqs, labs = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True)
    return padded, torch.tensor(labs)

train_loader = DataLoader(
    SeqDataset(X_train, y_train),
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    SeqDataset(X_val, y_val),
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

# -------------------------
# 7. GRU Encoder
# -------------------------
class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return h.squeeze(0)

# -------------------------
# 8. DQN Agent
# -------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.net(state)

# -------------------------
# 9. Training Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = GRUEncoder(len(word2idx)).to(device)
dqn = DQN(128, action_dim=len(set(labels))).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(dqn.parameters()),
    lr=0.001
)
loss_fn = nn.MSELoss()

def reward_fn(action, label):
    return 10 if action == label else -5

# -------------------------
# 10. Training Loop
# -------------------------
EPOCHS = 20
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    encoder.train()
    dqn.train()

    epoch_loss = 0
    y_true, y_pred = [], []

    for seqs, labs in train_loader:
        seqs, labs = seqs.to(device), labs.to(device)

        state = encoder(seqs)
        qvals = dqn(state)
        actions = torch.argmax(qvals, dim=1)

        rewards = torch.tensor(
            [reward_fn(a.item(), l.item()) for a, l in zip(actions, labs)],
            device=device, dtype=torch.float
        )

        target = qvals.clone().detach()
        for i in range(len(actions)):
            target[i, actions[i]] = rewards[i]

        loss = loss_fn(qvals, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        y_true.extend(labs.cpu().numpy())
        y_pred.extend(actions.cpu().numpy())

    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(accuracy_score(y_true, y_pred))

    # Validation
    encoder.eval()
    dqn.eval()
    val_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for seqs, labs in val_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            state = encoder(seqs)
            qvals = dqn(state)
            actions = torch.argmax(qvals, dim=1)

            val_loss += loss_fn(qvals, qvals).item()
            y_true.extend(labs.cpu().numpy())
            y_pred.extend(actions.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    val_accs.append(accuracy_score(y_true, y_pred))

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f} | "
        f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}"
    )

# -------------------------
# 11. Visualization
# -------------------------
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

plt.figure()
plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Curve")
plt.show()

# -------------------------
# 12. Evaluation Metrics
# -------------------------
encoder.eval()
dqn.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for seqs, labs in val_loader:
        seqs, labs = seqs.to(device), labs.to(device)
        state = encoder(seqs)
        qvals = dqn(state)
        actions = torch.argmax(qvals, dim=1)

        y_true.extend(labs.cpu().numpy())
        y_pred.extend(actions.cpu().numpy())

print("\nEvaluation Metrics:")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_true, y_pred, average="weighted", zero_division=0))
print("F1-score :", f1_score(y_true, y_pred, average="weighted", zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# -------------------------
# 13. Testing / Inference
# -------------------------
def predict_sequence(seq):
    encoder.eval()
    dqn.eval()

    tokens = [word2idx.get(str(x), 0) for x in seq]
    tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        state = encoder(tensor)
        qvals = dqn(state)
        action = torch.argmax(qvals).item()

    return action

print("\nSample Test Prediction:")
print("Predicted Action:", predict_sequence(X_val[0]))
print("True Label      :", y_val[0])
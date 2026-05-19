from pathlib import Path
import ast
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import timm


DATA = Path("data")

OUT = Path("outputs/exp003_effb0_weighted")
OUT.mkdir(parents=True, exist_ok=True)

MEL_DIR = Path("mels/train_audio")

OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
DURATION = 5
TARGET_LEN = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 16000

BATCH_SIZE = 128
EPOCHS = 10
LR = 2e-4
NUM_WORKERS = 4
SEED = 42

def filename_to_npy_path(filename):
    filename = str(filename).strip()
    return MEL_DIR / Path(filename).with_suffix(".npy")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_logmel(path, train=True):
    y, _ = librosa.load(path, sr=SR, mono=True)

    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        if train:
            start = np.random.randint(0, len(y) - TARGET_LEN + 1)
        else:
            start = max(0, (len(y) - TARGET_LEN) // 2)
        y = y[start:start + TARGET_LEN]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)


class BirdDataset(Dataset):
    def __init__(self, df, target_cols, label_to_idx, train=True):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.label_to_idx = label_to_idx
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = filename_to_npy_path(row["filename"])

        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed mel: {path}")

        x = np.load(path).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)  # [1, mel, time]

        y = torch.zeros(len(self.target_cols), dtype=torch.float32)

        primary = str(row["primary_label"])
        if primary in self.label_to_idx:
            y[self.label_to_idx[primary]] = 1.0

        # Include secondary labels if present and valid.
        try:
            secondary = ast.literal_eval(row["secondary_labels"])
            for lab in secondary:
                lab = str(lab)
                if lab in self.label_to_idx:
                    y[self.label_to_idx[lab]] = 1.0
        except Exception:
            pass

        return x, y


def build_model(num_classes):
    model = timm.create_model(
        "tf_efficientnet_b0_ns",
        pretrained=True,
        in_chans=1,
        num_classes=num_classes,
    )
    return model


def macro_auc(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        # roc_auc_score fails if class has only one label in val.
        if len(np.unique(y_true[:, i])) < 2:
            continue
        scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(scores)) if scores else float("nan")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y = []
    all_p = []

    for x, y in tqdm(loader, desc="valid", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * x.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    auc = macro_auc(y_true, y_pred)

    return total_loss / len(loader.dataset), auc

def make_weighted_sampler(df):
    """
    Create sample weights based on primary_label frequency.

    Uses 1 / sqrt(class_count) rather than 1 / class_count
    so rare classes get sampled more often without letting
    one-example classes completely dominate.
    """
    labels = df["primary_label"].astype(str).str.strip()
    counts = labels.value_counts()

    weights = labels.map(lambda lab: 1.0 / np.sqrt(counts[lab])).values
    weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    return sampler

def main():
    seed_everything(SEED)

    train_df = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_cols = sample.columns[1:].astype(str).tolist()
    label_to_idx = {lab: i for i, lab in enumerate(target_cols)}

    # Simple stratified split by primary label.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_df["fold"] = -1

    for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df["primary_label"])):
        train_df.loc[val_idx, "fold"] = fold

    fold = 0
    trn = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val = train_df[train_df["fold"] == fold].reset_index(drop=True)

    #confirm weights make sense
    print("\nTrain class counts before weighting:")
    print(trn["primary_label"].value_counts().describe())

    tmp_counts = trn["primary_label"].value_counts()
    print("\nExample class weights:")
    for lab in tmp_counts.head(5).index:
        print(lab, "count=", tmp_counts[lab], "weight=", 1.0 / np.sqrt(tmp_counts[lab]))

    for lab in tmp_counts.tail(5).index:
        print(lab, "count=", tmp_counts[lab], "weight=", 1.0 / np.sqrt(tmp_counts[lab]))

    print("train:", trn.shape, "valid:", val.shape)
    print("num targets:", len(target_cols))

    train_ds = BirdDataset(trn, target_cols, label_to_idx, train=True)
    val_ds = BirdDataset(val, target_cols, label_to_idx, train=False)

    train_sampler = make_weighted_sampler(trn)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_model(num_classes=len(target_cols)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_auc = -1

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_auc={val_auc:.5f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "target_cols": target_cols,
                    "sr": SR,
                    "duration": DURATION,
                    "n_mels": N_MELS,
                },
                OUT / "fold0_best.pt",
            )
            print("saved best:", OUT / "fold0_best.pt")


if __name__ == "__main__":
    main()
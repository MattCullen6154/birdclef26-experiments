from pathlib import Path
import ast
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm


DATA = Path("data")

OUT = Path("outputs/exp004_effb0_soundscapes")
OUT.mkdir(parents=True, exist_ok=True)

MEL_DIR = Path("mels/train_audio")
SOUNDSCAPE_META = Path("mels/train_soundscapes_metadata.csv")

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

    def encode_labels(self, row):
        y = torch.zeros(len(self.target_cols), dtype=torch.float32)

        source = row.get("source", "train_audio")

        if source == "soundscape":
            # Soundscape labels are semicolon-separated strings:
            # "22961;23158;24321;517063;65380"
            labels = str(row["primary_label"]).strip().split(";")
            for lab in labels:
                lab = str(lab).strip()
                if lab in self.label_to_idx:
                    y[self.label_to_idx[lab]] = 1.0

        else:
            primary = str(row["primary_label"]).strip()
            if primary in self.label_to_idx:
                y[self.label_to_idx[primary]] = 1.0

            try:
                secondary = ast.literal_eval(row["secondary_labels"])
                for lab in secondary:
                    lab = str(lab).strip()
                    if lab in self.label_to_idx:
                        y[self.label_to_idx[lab]] = 1.0
            except Exception:
                pass

        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = Path(row["mel_path"])

        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed mel: {path}")

        x = np.load(path).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)  # [1, mel, time]

        y = self.encode_labels(row)

        return x, y


def filename_to_npy_path(filename):
    filename = str(filename).strip()
    return MEL_DIR / Path(filename).with_suffix(".npy")


def build_train_audio_metadata(train_df):
    df = train_df.copy()
    df["source"] = "train_audio"
    df["filename"] = df["filename"].astype(str).str.strip()
    df["primary_label"] = df["primary_label"].astype(str).str.strip()
    df["secondary_labels"] = df["secondary_labels"].fillna("[]")
    df["mel_path"] = df["filename"].apply(lambda x: str(filename_to_npy_path(x)))
    return df

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


def main():
    seed_everything(SEED)

    train_df = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_cols = sample.columns[1:].astype(str).tolist()
    label_to_idx = {lab: i for i, lab in enumerate(target_cols)}

    # Simple stratified split by primary label.
    # Clean and attach mel paths for focal train_audio.
    train_df = build_train_audio_metadata(train_df)

    # Same focal split as exp002, so comparison is clean.
    counts = train_df["primary_label"].value_counts()
    rare_labels = counts[counts < 2].index
    common_labels = counts[counts >= 2].index

    rare_df = train_df[train_df["primary_label"].isin(rare_labels)].copy()
    common_df = train_df[train_df["primary_label"].isin(common_labels)].copy()

    trn_common, val = train_test_split(
        common_df,
        test_size=0.2,
        random_state=SEED,
        stratify=common_df["primary_label"],
    )

    trn_audio = (
        pd.concat([trn_common, rare_df], axis=0)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    val = val.reset_index(drop=True)

    # Add labeled soundscape windows to training only.
    soundscape_df = pd.read_csv(SOUNDSCAPE_META)
    soundscape_df["source"] = "soundscape"
    soundscape_df["primary_label"] = soundscape_df["primary_label"].astype(str).str.strip()
    soundscape_df["secondary_labels"] = "[]"

    # Keep only soundscape labels that contain at least one known target.
    def has_known_label(label_string):
        labels = str(label_string).split(";")
        return any(lab.strip() in label_to_idx for lab in labels)

    soundscape_df = soundscape_df[soundscape_df["primary_label"].apply(has_known_label)].copy()

    trn = (
        pd.concat([trn_audio, soundscape_df], axis=0, ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    print("train_audio train:", trn_audio.shape)
    print("soundscape train:", soundscape_df.shape)
    print("combined train:", trn.shape)
    print("valid focal:", val.shape)
    print("num targets:", len(target_cols))

    train_ds = BirdDataset(trn, target_cols, label_to_idx, train=True)
    val_ds = BirdDataset(val, target_cols, label_to_idx, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
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
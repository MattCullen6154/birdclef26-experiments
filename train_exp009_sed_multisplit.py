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

OUT_ROOT = Path("outputs/exp009_sed_multisplit")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


MEL_DIR = Path("mels/train_audio")
SOUNDSCAPE_META = Path("mels/train_soundscapes_metadata.csv")


SR = 32000
DURATION = 5
TARGET_LEN = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 16000

BATCH_SIZE = 64
SPLIT_SEEDS = [42, 1337]#, 2026, 7, 99]
WEIGHT_DECAY = 1e-4
LR = 1e-4
EPOCHS = 8
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

def split_soundscapes_by_filename(soundscape_df, val_frac=0.2, seed=SEED):
    """
    Split soundscape labels by filename, not by row.

    This prevents leakage where adjacent 5-second windows from the same
    recording appear in both train and validation.
    """
    filenames = soundscape_df["filename"].astype(str).str.strip().unique()

    train_files, val_files = train_test_split(
        filenames,
        test_size=val_frac,
        random_state=seed,
        shuffle=True,
    )

    train_files = set(train_files)
    val_files = set(val_files)

    ss_train = soundscape_df[soundscape_df["filename"].isin(train_files)].copy()
    ss_val = soundscape_df[soundscape_df["filename"].isin(val_files)].copy()

    return ss_train, ss_val

def macro_auc_with_count(y_true, y_pred):
    scores = []
    scored_classes = 0

    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue

        scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        scored_classes += 1

    auc = float(np.mean(scores)) if scores else float("nan")
    return auc, scored_classes

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
        x = torch.tensor(x).unsqueeze(0)

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

class EfficientNetSED(nn.Module):
    """
    EfficientNet backbone + simple SED-style temporal head.

    Input:
      x: [B, 1, n_mels, time]

    Backbone output:
      features: [B, C, H, W]

    We average over frequency H, keep time W:
      time_features: [B, C, W]

    Then use a 1x1 Conv1d over channels to produce frame-level class logits:
      frame_logits: [B, num_classes, W]

    Clip-level logits combine:
      mean over time: stable broad evidence
      max over time: brief event evidence
    """

    def __init__(
        self,
        num_classes,
        model_name="tf_efficientnet_b0_ns",
        pretrained=True,
        mean_weight=0.5,
        max_weight=0.5,
        dropout=0.3,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )

        self.num_features = self.backbone.num_features
        self.mean_weight = mean_weight
        self.max_weight = max_weight

        self.dropout = nn.Dropout(dropout)

        # Conv1d maps [B, C, T] -> [B, num_classes, T]
        self.frame_head = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)  # [B, C, H, W]

        # Average over frequency dimension, keep time dimension.
        time_features = features.mean(dim=2)          # [B, C, W]
        time_features = self.dropout(time_features)

        frame_logits = self.frame_head(time_features) # [B, classes, W]

        mean_logits = frame_logits.mean(dim=2)        # [B, classes]
        max_logits = frame_logits.max(dim=2).values   # [B, classes]

        clip_logits = (
            self.mean_weight * mean_logits
            + self.max_weight * max_logits
        )


        return clip_logits


def build_model(num_classes):
    return EfficientNetSED(
        num_classes=num_classes,
        model_name="tf_efficientnet_b0_ns",
        pretrained=True,
        mean_weight=0.5,
        max_weight=0.5,
        dropout=0.2,
    )


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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * x.size(0)
        all_y.append(y.cpu().numpy())
        all_p.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_p, axis=0)

    auc, scored_classes = macro_auc_with_count(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, auc, scored_classes

def run_one_split(split_seed):
    print("\n" + "=" * 80)
    print(f"Running soundscape split seed: {split_seed}")
    print("=" * 80)

    out_dir = OUT_ROOT / f"seed_{split_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(split_seed)

    train_df = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_cols = sample.columns[1:].astype(str).tolist()
    label_to_idx = {lab: i for i, lab in enumerate(target_cols)}

    # All focal audio is used for training.
    train_audio_df = build_train_audio_metadata(train_df)

    # Load soundscape metadata.
    soundscape_df = pd.read_csv(SOUNDSCAPE_META)
    soundscape_df["source"] = "soundscape"
    soundscape_df["filename"] = soundscape_df["filename"].astype(str).str.strip()
    soundscape_df["primary_label"] = soundscape_df["primary_label"].astype(str).str.strip()
    soundscape_df["secondary_labels"] = "[]"

    def has_known_label(label_string):
        labels = str(label_string).split(";")
        return any(lab.strip() in label_to_idx for lab in labels)

    soundscape_df = soundscape_df[
        soundscape_df["primary_label"].apply(has_known_label)
    ].copy()

    ss_train, ss_val = split_soundscapes_by_filename(
        soundscape_df,
        val_frac=0.2,
        seed=split_seed,
    )

    trn = (
        pd.concat([train_audio_df, ss_train], axis=0, ignore_index=True)
        .sample(frac=1, random_state=split_seed)
        .reset_index(drop=True)
    )

    val = ss_val.reset_index(drop=True)

    print("train_audio train:", train_audio_df.shape)
    print("soundscape train:", ss_train.shape)
    print("soundscape valid:", ss_val.shape)
    print("combined train:", trn.shape)
    print("unique train soundscape files:", ss_train["filename"].nunique())
    print("unique valid soundscape files:", ss_val["filename"].nunique())
    print("num targets:", len(target_cols))

    train_ds = BirdDataset(trn, target_cols, label_to_idx, train=True)
    val_ds = BirdDataset(val, target_cols, label_to_idx, train=False)

    """
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
    """

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_model(num_classes=len(target_cols)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=2e-5,
    )

    best_auc = -1
    best_epoch = -1
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc, scored_classes = validate(model, val_loader, criterion, device)

        row = {
            "split_seed": split_seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "scored_classes": scored_classes,
            "soundscape_train_rows": len(ss_train),
            "soundscape_val_rows": len(ss_val),
            "soundscape_train_files": ss_train["filename"].nunique(),
            "soundscape_val_files": ss_val["filename"].nunique(),
        }
        history.append(row)

        print(
            f"Seed {split_seed} Epoch {epoch}: "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"val_auc={val_auc:.5f} "
            f"scored_classes={scored_classes}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(
            {
                "model": model.state_dict(),
                "target_cols": target_cols,
                "sr": SR,
                "duration": DURATION,
                "n_mels": N_MELS,
                "split_seed": split_seed,
                "best_epoch": best_epoch,
                "best_auc": best_auc,
                "model_name": "tf_efficientnet_b0_ns",
                "model_type": "efficientnet_sed",
                "mean_weight": 0.5,
                "max_weight": 0.5,
                "dropout": 0.2,
            },
            out_dir / "fold0_best.pt",
            )
            print("saved best:", out_dir / "fold0_best.pt")

            scheduler.step()

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "history.csv", index=False)

    summary = {
        "split_seed": split_seed,
        "best_epoch": best_epoch,
        "best_auc": best_auc,
        "best_checkpoint": str(out_dir / "fold0_best.pt"),
        "soundscape_train_rows": len(ss_train),
        "soundscape_val_rows": len(ss_val),
        "soundscape_train_files": ss_train["filename"].nunique(),
        "soundscape_val_files": ss_val["filename"].nunique(),
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary

def main():
    all_summaries = []

    for split_seed in SPLIT_SEEDS:
        summary = run_one_split(split_seed)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT_ROOT / "summary.csv", index=False)

    print("\n" + "=" * 80)
    print("MULTI-SPLIT SUMMARY")
    print("=" * 80)
    print(summary_df)
    print()
    print("mean best_auc:", summary_df["best_auc"].mean())
    print("std best_auc:", summary_df["best_auc"].std())
    print("mean best_epoch:", summary_df["best_epoch"].mean())


if __name__ == "__main__":
    main()
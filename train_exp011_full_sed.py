from pathlib import Path
import ast
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import timm


DATA = Path("data")
OUT = Path("outputs/exp011_full_sed")
OUT.mkdir(parents=True, exist_ok=True)



MEL_DIR = Path("mels/train_audio")
SOUNDSCAPE_META = Path("mels/train_soundscapes_metadata.csv")

SR = 32000
DURATION = 5
N_MELS = 128

BATCH_SIZE = 64
EPOCHS = 8
LR = 2e-4
NUM_WORKERS = 4
SEED = 42


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    We average over frequency H and keep time W:
      time_features: [B, C, W]

    Then produce frame-level class logits:
      frame_logits: [B, num_classes, W]

    Clip-level logits combine:
      mean over time: broad stable evidence
      max over time: brief event evidence
    """

    def __init__(
        self,
        num_classes,
        model_name="tf_efficientnet_b0_ns",
        pretrained=True,
        mean_weight=0.5,
        max_weight=0.5,
        dropout=0.2,
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

        self.frame_head = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)  # [B, C, H, W]

        # Average over frequency, preserve time.
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main():
    seed_everything(SEED)

    train_df = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_cols = sample.columns[1:].astype(str).tolist()
    label_to_idx = {lab: i for i, lab in enumerate(target_cols)}

    # All focal train_audio clips.
    train_audio_df = build_train_audio_metadata(train_df)

    # All labeled soundscape windows.
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

    trn = (
        pd.concat([train_audio_df, soundscape_df], axis=0, ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    print("train_audio:", train_audio_df.shape)
    print("soundscape:", soundscape_df.shape)
    print("combined train:", trn.shape)
    print("num targets:", len(target_cols))

    print("\nSource counts:")
    print(trn["source"].value_counts())

    print("\nExample rows:")
    print(trn[["source", "filename", "primary_label", "mel_path"]].head())

    train_ds = BirdDataset(trn, target_cols, label_to_idx, train=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\ndevice:", device)

    model = build_model(num_classes=len(target_cols)).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        print(f"Epoch {epoch}: train_loss={train_loss:.5f}")

        ckpt = {
            "model": model.state_dict(),
            "target_cols": target_cols,
            "sr": SR,
            "duration": DURATION,
            "n_mels": N_MELS,
            "epoch": epoch,
            "train_loss": train_loss,
            "model_name": "tf_efficientnet_b0_ns",
            "model_type": "efficientnet_sed",
            "mean_weight": 0.5,
            "max_weight": 0.5,
            "dropout": 0.2,
            "experiment": "exp011_full_sed",
            "preprocessing": {
                "mel_source_train_audio": str(MEL_DIR),
                "mel_source_soundscapes": str(SOUNDSCAPE_META),
                "sr": SR,
                "duration": DURATION,
                "n_mels": N_MELS,
            },
        }

        epoch_path = OUT / f"epoch{epoch:02d}.pt"

        torch.save(ckpt, epoch_path)
        torch.save(ckpt, OUT / "last.pt")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "checkpoint": str(epoch_path),
            }
        )

        pd.DataFrame(history).to_csv(OUT / "history.csv", index=False)

        print("saved:", epoch_path)
        print("saved:", OUT / "last.pt")

    print("\nDone.")
    print("Primary candidate based on multisplit soundscape validation:")
    print(OUT / "epoch06.pt")
    print("Backup candidate:")
    print(OUT / "epoch05.pt")


if __name__ == "__main__":
    main()
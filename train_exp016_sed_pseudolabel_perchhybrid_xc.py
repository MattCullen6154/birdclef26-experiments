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
OUT = Path("outputs/exp016_sed_pseudolabel_perchhybrid_xc")
OUT.mkdir(parents=True, exist_ok=True)


MEL_DIR = Path("mels/train_audio")
SOUNDSCAPE_META = Path("mels/train_soundscapes_metadata.csv")
PSEUDO_PROBS = Path("mels/train_soundscapes_pseudo_probs_exp015_perch_hybrid.npy")
PSEUDO_META = Path("mels/train_soundscapes_pseudo_metadata.csv")

XC_META = Path("mels/xenocanto_teacher_filtered_metadata.csv")
XC_PROBS = Path("mels/xenocanto_teacher_filtered_probs.npy")

PSEUDO_LOSS_WEIGHT = 0.15
XC_LOSS_WEIGHT = 0.10

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
        features = self.backbone.forward_features(x)
        time_features = features.mean(dim=2)
        time_features = self.dropout(time_features)

        frame_logits = self.frame_head(time_features)

        mean_logits = frame_logits.mean(dim=2)
        max_logits = frame_logits.max(dim=2).values

        return self.mean_weight * mean_logits + self.max_weight * max_logits


def build_model(num_classes):
    return EfficientNetSED(
        num_classes=num_classes,
        pretrained=True,
        mean_weight=0.5,
        max_weight=0.5,
        dropout=0.2,
    )


class BirdDataset(Dataset):
    def __init__(self, df, target_cols, label_to_idx, pseudo_probs=None):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.label_to_idx = label_to_idx
        self.pseudo_probs = pseudo_probs

    def __len__(self):
        return len(self.df)

    def encode_hard_labels(self, row):
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
                import ast
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
            raise FileNotFoundError(f"Missing mel: {path}")

        x = np.load(path).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)

        source = row.get("source", "train_audio")

        if str(source).startswith("pseudo"):
            pseudo_idx = int(row["pseudo_idx"])
            y = torch.tensor(self.pseudo_probs[pseudo_idx], dtype=torch.float32)

            if source == "pseudo_xenocanto":
                w = torch.tensor(XC_LOSS_WEIGHT, dtype=torch.float32)
            else:
                w = torch.tensor(PSEUDO_LOSS_WEIGHT, dtype=torch.float32)
        else:
            y = self.encode_hard_labels(row)
            w = torch.tensor(1.0, dtype=torch.float32)

        return x, y, w


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_weight = 0.0

    for x, y, w in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss_matrix = criterion(logits, y)      # [B, C]
        loss_per_sample = loss_matrix.mean(dim=1)  # [B]
        loss = (loss_per_sample * w).sum() / (w.sum() + 1e-6)

        loss.backward()
        optimizer.step()

        total_loss += float((loss_per_sample * w).sum().item())
        total_weight += float(w.sum().item())

    return total_loss / max(total_weight, 1e-6)


def main():
    seed_everything(SEED)

    train_df = pd.read_csv(DATA / "train.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")

    target_cols = sample.columns[1:].astype(str).tolist()
    label_to_idx = {lab: i for i, lab in enumerate(target_cols)}

    # Hard-labeled focal train_audio.
    train_audio_df = build_train_audio_metadata(train_df)

    # Hard-labeled official soundscape windows.
    soundscape_df = pd.read_csv(SOUNDSCAPE_META)
    soundscape_df["source"] = "soundscape"
    soundscape_df["filename"] = soundscape_df["filename"].astype(str).str.strip()
    soundscape_df["primary_label"] = soundscape_df["primary_label"].astype(str).str.strip()
    soundscape_df["secondary_labels"] = "[]"
    soundscape_df["mel_path"] = soundscape_df["mel_path"].astype(str)

    def has_known_label(label_string):
        labels = str(label_string).split(";")
        return any(lab.strip() in label_to_idx for lab in labels)

    soundscape_df = soundscape_df[
        soundscape_df["primary_label"].apply(has_known_label)
    ].copy()

    # Soft pseudo-labeled unlabeled soundscape windows.
    pseudo_df = pd.read_csv(PSEUDO_META)
    pseudo_probs = np.load(PSEUDO_PROBS)

    pseudo_df["source"] = "pseudo_soundscape"
    pseudo_df["filename"] = pseudo_df["filename"].astype(str).str.strip()
    pseudo_df["primary_label"] = ""
    pseudo_df["secondary_labels"] = "[]"
    pseudo_df["mel_path"] = pseudo_df["mel_path"].astype(str)
    pseudo_df["pseudo_idx"] = pseudo_df["pseudo_idx"].astype(int)

    # New Xeno-canto pseudo rows
    xc_df = pd.read_csv(XC_META)
    xc_probs = np.load(XC_PROBS)

    xc_df["source"] = "pseudo_xenocanto"
    xc_df["filename"] = xc_df["filename"].astype(str).str.strip()
    xc_df["primary_label"] = xc_df["target_label"].astype(str)
    xc_df["secondary_labels"] = "[]"
    xc_df["mel_path"] = xc_df["mel_path"].astype(str)

    # IMPORTANT: these indices must come after the soundscape pseudo probs
    xc_df["pseudo_idx"] = np.arange(len(pseudo_df), len(pseudo_df) + len(xc_df), dtype=int)

    # Combine metadata and probabilities
    combined_pseudo_df = pd.concat([pseudo_df, xc_df], ignore_index=True)
    combined_pseudo_probs = np.concatenate([pseudo_probs, xc_probs], axis=0).astype(np.float32)

    assert len(combined_pseudo_df) == len(combined_pseudo_probs)
    assert combined_pseudo_df["pseudo_idx"].min() == 0
    assert combined_pseudo_df["pseudo_idx"].max() == len(combined_pseudo_df) - 1

    trn = (
        pd.concat([train_audio_df, soundscape_df, combined_pseudo_df], axis=0, ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

    print("train_audio:", train_audio_df.shape)
    print("soundscape:", soundscape_df.shape)
    print("pseudo soundscape:", pseudo_df.shape)
    print("pseudo xenocanto:", xc_df.shape)
    print("combined pseudo:", combined_pseudo_df.shape)
    print("combined train:", trn.shape)
    print("num targets:", len(target_cols))
    print("pseudo loss weight:", PSEUDO_LOSS_WEIGHT)
    print("xc loss weight:", XC_LOSS_WEIGHT)

    print("\nSource counts:")
    print(trn["source"].value_counts())

    train_ds = BirdDataset(
        trn,
        target_cols,
        label_to_idx,
        pseudo_probs=combined_pseudo_probs,
    )

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

    criterion = nn.BCEWithLogitsLoss(reduction="none")
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
            "experiment": "exp016_sed_pseudolabel_perchhybrid_xc",
            "mean_weight": 0.5,
            "max_weight": 0.5,
            "dropout": 0.2,
            "pseudo_loss_weight": PSEUDO_LOSS_WEIGHT,
            "pseudo_probs": str(PSEUDO_PROBS),
            "teacher": "hybrid: base 0.80*exp011/exp014 SED+0.20*clip, Perch blended on mapped classes",
            "xc_loss_weight": XC_LOSS_WEIGHT,
            "xc_meta": str(XC_META),
            "xc_probs": str(XC_PROBS),
            "xc_rows": int(len(xc_df)),
            "combined_pseudo_rows": int(len(combined_pseudo_df)),
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
    print("Candidate checkpoints:")
    print(OUT / "epoch05.pt")
    print(OUT / "epoch06.pt")


if __name__ == "__main__":
    main()
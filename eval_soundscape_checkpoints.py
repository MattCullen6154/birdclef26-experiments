from pathlib import Path
import ast
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm


DATA = Path("data")
SOUNDSCAPE_META = Path("mels/train_soundscapes_metadata.csv")

SR = 32000
DURATION = 5
N_MELS = 128

BATCH_SIZE = 128
NUM_WORKERS = 4
SEED = 42

CHECKPOINTS = {
    "exp006_epoch05": Path("outputs/exp006_full_soundscapes/epoch05.pt"),
    "exp006_epoch06": Path("outputs/exp006_full_soundscapes/epoch06.pt"),

    "exp008_epoch03": Path("outputs/exp008_background/epoch03.pt"),
    "exp008_epoch04": Path("outputs/exp008_background/epoch04.pt"),
    "exp008_epoch05": Path("outputs/exp008_background/epoch05.pt"),
    "exp008_epoch06": Path("outputs/exp008_background/epoch06.pt"),
    "exp008_epoch07": Path("outputs/exp008_background/epoch07.pt"),
    "exp008_epoch08": Path("outputs/exp008_background/epoch08.pt"),
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes):
    model = timm.create_model(
        "tf_efficientnet_b0_ns",
        pretrained=False,
        in_chans=1,
        num_classes=num_classes,
    )
    return model


def split_soundscapes_by_filename(soundscape_df, val_frac=0.2, seed=SEED):
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


class SoundscapeDataset(Dataset):
    def __init__(self, df, target_cols, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def encode_labels(self, label_string):
        y = torch.zeros(len(self.target_cols), dtype=torch.float32)

        labels = str(label_string).strip().split(";")
        for lab in labels:
            lab = str(lab).strip()
            if lab in self.label_to_idx:
                y[self.label_to_idx[lab]] = 1.0

        return y

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = Path(row["mel_path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing precomputed mel: {path}")

        x = np.load(path).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)

        y = self.encode_labels(row["primary_label"])

        return x, y


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_y = []
    all_p = []

    for x, y in tqdm(loader, desc="eval", leave=False):
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


def load_checkpoint_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    target_cols = ckpt["target_cols"]
    model = build_model(num_classes=len(target_cols))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    return model, target_cols


def main():
    seed_everything(SEED)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    target_cols_from_sample = sample.columns[1:].astype(str).tolist()
    label_to_idx = {lab: i for i, lab in enumerate(target_cols_from_sample)}

    soundscape_df = pd.read_csv(SOUNDSCAPE_META)
    soundscape_df["source"] = "soundscape"
    soundscape_df["filename"] = soundscape_df["filename"].astype(str).str.strip()
    soundscape_df["primary_label"] = soundscape_df["primary_label"].astype(str).str.strip()

    def has_known_label(label_string):
        labels = str(label_string).split(";")
        return any(lab.strip() in label_to_idx for lab in labels)

    soundscape_df = soundscape_df[soundscape_df["primary_label"].apply(has_known_label)].copy()

    ss_train, ss_val = split_soundscapes_by_filename(
        soundscape_df,
        val_frac=0.2,
        seed=SEED,
    )

    print("soundscape train rows:", ss_train.shape)
    print("soundscape val rows:", ss_val.shape)
    print("unique train files:", ss_train["filename"].nunique())
    print("unique val files:", ss_val["filename"].nunique())
    print("val label examples:")
    print(ss_val[["filename", "start", "end", "primary_label", "mel_path"]].head())

    val_ds = SoundscapeDataset(ss_val, target_cols_from_sample, label_to_idx)

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    results = []

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n=== Evaluating {name} ===")
        print("checkpoint:", ckpt_path)

        if not ckpt_path.exists():
            print("MISSING checkpoint, skipping")
            continue

        model, target_cols_from_ckpt = load_checkpoint_model(ckpt_path, device)

        if target_cols_from_ckpt != target_cols_from_sample:
            raise ValueError(f"Target column mismatch for {name}")

        val_loss, val_auc, scored_classes = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        result = {
            "name": name,
            "checkpoint": str(ckpt_path),
            "val_loss": val_loss,
            "val_auc": val_auc,
            "scored_classes": scored_classes,
        }
        results.append(result)

        print(
            f"{name}: "
            f"val_loss={val_loss:.5f} "
            f"val_auc={val_auc:.5f} "
            f"scored_classes={scored_classes}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(results_df.sort_values("val_auc", ascending=False))

    out_path = Path("outputs/soundscape_checkpoint_eval.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
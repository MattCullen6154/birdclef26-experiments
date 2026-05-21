from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import timm


DATA = Path("data")
PSEUDO_META = Path("mels/train_soundscapes_pseudo_metadata.csv")
OUT_PROBS = Path("mels/train_soundscapes_pseudo_probs_exp006_e5e6.npy")

CKPT_PATHS = [
    Path("outputs/exp006_full_soundscapes/epoch05.pt"),
    Path("outputs/exp006_full_soundscapes/epoch06.pt"),
]

BATCH_SIZE = 128
NUM_WORKERS = 4


def build_model(num_classes):
    model = timm.create_model(
        "tf_efficientnet_b0_ns",
        pretrained=False,
        in_chans=1,
        num_classes=num_classes,
    )
    return model


class PseudoMelDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["mel_path"])
        x = np.load(path).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)
        return x


@torch.no_grad()
def predict_ensemble(models, loader, device):
    all_probs = []

    for x in tqdm(loader, desc="teacher pseudo inference"):
        x = x.to(device, non_blocking=True)

        probs_sum = None

        for model in models:
            logits = model(x)
            probs = torch.sigmoid(logits)

            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum = probs_sum + probs

        probs_avg = probs_sum / len(models)
        all_probs.append(probs_avg.cpu().numpy().astype(np.float32))

    return np.concatenate(all_probs, axis=0)


def main():
    meta = pd.read_csv(PSEUDO_META)
    meta = meta.sort_values("pseudo_idx").reset_index(drop=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    target_cols = sample.columns[1:].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    models = []

    for ckpt_path in CKPT_PATHS:
        print("loading:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        if ckpt["target_cols"] != target_cols:
            raise ValueError(f"target mismatch: {ckpt_path}")

        model = build_model(num_classes=len(target_cols))
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        models.append(model)

    ds = PseudoMelDataset(meta)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    probs = predict_ensemble(models, loader, device)

    print("probs shape:", probs.shape)
    print("min:", probs.min(), "max:", probs.max(), "mean:", probs.mean())

    OUT_PROBS.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PROBS, probs)
    print("saved:", OUT_PROBS)


if __name__ == "__main__":
    main()
    
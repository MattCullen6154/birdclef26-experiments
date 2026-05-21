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
OUT_PROBS = Path("mels/train_soundscapes_pseudo_probs_exp012_sedclip.npy")

SED_CKPT_PATHS = [
    Path("outputs/exp011_full_sed/epoch06.pt"),
]

CLIP_CKPT_PATHS = [
    Path("outputs/exp006_full_soundscapes/epoch05.pt"),
    Path("outputs/exp006_full_soundscapes/epoch06.pt"),
]

SED_WEIGHT = 0.95
CLIP_WEIGHT = 0.05

BATCH_SIZE = 128
NUM_WORKERS = 4


class EfficientNetSED(nn.Module):
    def __init__(
        self,
        num_classes,
        model_name="tf_efficientnet_b0_ns",
        pretrained=False,
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
        time_features = features.mean(dim=2)          # [B, C, W]
        time_features = self.dropout(time_features)

        frame_logits = self.frame_head(time_features) # [B, classes, W]

        mean_logits = frame_logits.mean(dim=2)
        max_logits = frame_logits.max(dim=2).values

        return self.mean_weight * mean_logits + self.max_weight * max_logits


def build_sed_model(num_classes):
    return EfficientNetSED(
        num_classes=num_classes,
        pretrained=False,
        mean_weight=0.5,
        max_weight=0.5,
        dropout=0.2,
    )


def build_clip_model(num_classes):
    return timm.create_model(
        "tf_efficientnet_b0_ns",
        pretrained=False,
        in_chans=1,
        num_classes=num_classes,
    )


class PseudoMelDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["mel_path"])

        if not path.exists():
            raise FileNotFoundError(f"Missing mel: {path}")

        x = np.load(path).astype(np.float32)
        x = torch.tensor(x).unsqueeze(0)  # [1, n_mels, time]
        return x


@torch.no_grad()
def predict_ensemble(models, x):
    probs_sum = None

    for model in models:
        logits = model(x)
        probs = torch.sigmoid(logits)

        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum = probs_sum + probs

    return probs_sum / len(models)


@torch.no_grad()
def predict_teacher(sed_models, clip_models, loader, device):
    all_probs = []

    for x in tqdm(loader, desc="teacher pseudo inference"):
        x = x.to(device, non_blocking=True)

        sed_probs = predict_ensemble(sed_models, x)
        clip_probs = predict_ensemble(clip_models, x)

        probs = SED_WEIGHT * sed_probs + CLIP_WEIGHT * clip_probs
        all_probs.append(probs.cpu().numpy().astype(np.float32))

    return np.concatenate(all_probs, axis=0)


def load_models(ckpt_paths, build_fn, target_cols, device, label):
    models = []

    for ckpt_path in ckpt_paths:
        print(f"loading {label}:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)

        if ckpt["target_cols"] != target_cols:
            raise ValueError(f"target mismatch: {ckpt_path}")

        print("  experiment:", ckpt.get("experiment"))
        print("  model_type:", ckpt.get("model_type"))
        print("  epoch:", ckpt.get("epoch"))

        model = build_fn(num_classes=len(target_cols))
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        models.append(model)

    return models


def main():
    meta = pd.read_csv(PSEUDO_META)
    meta = meta.sort_values("pseudo_idx").reset_index(drop=True)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    target_cols = sample.columns[1:].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("pseudo rows:", len(meta))
    print("teacher blend:", SED_WEIGHT, "SED +", CLIP_WEIGHT, "clip")

    sed_models = load_models(
        SED_CKPT_PATHS,
        build_sed_model,
        target_cols,
        device,
        label="SED",
    )

    clip_models = load_models(
        CLIP_CKPT_PATHS,
        build_clip_model,
        target_cols,
        device,
        label="clip",
    )

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

    probs = predict_teacher(sed_models, clip_models, loader, device)

    print("probs shape:", probs.shape)
    print("min:", probs.min(), "max:", probs.max(), "mean:", probs.mean())

    OUT_PROBS.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PROBS, probs)
    print("saved:", OUT_PROBS)


if __name__ == "__main__":
    main()
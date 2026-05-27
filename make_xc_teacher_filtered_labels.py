from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import timm


DATA = Path("data")
XC_META = Path("mels/xenocanto_pseudo_metadata.csv")

OUT_META = Path("mels/xenocanto_teacher_filtered_metadata.csv")
OUT_PROBS = Path("mels/xenocanto_teacher_filtered_probs.npy")

# Current best teacher:
# exp015 Perch-hybrid SED epoch06 + exp006 clip e5/e6, 80/20
SED_CKPT_PATHS = [
    Path("outputs/exp015_sed_pseudolabel_perchhybrid/epoch06.pt"),
]

CLIP_CKPT_PATHS = [
    Path("outputs/exp006_full_soundscapes/epoch05.pt"),
    Path("outputs/exp006_full_soundscapes/epoch06.pt"),
]

SED_WEIGHT = 0.80
CLIP_WEIGHT = 0.20

TARGET_THRESHOLD = 0.35

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


class XCMelDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["mel_path"]).astype(np.float32)
        return torch.tensor(x).unsqueeze(0), idx


def load_models(paths, build_fn, target_cols, device, label):
    models = []

    for p in paths:
        print(f"loading {label}:", p)
        ckpt = torch.load(p, map_location=device)

        ckpt_targets = ckpt["target_cols"]
        if ckpt_targets != target_cols:
            raise ValueError(f"target mismatch for {p}")

        print("  experiment:", ckpt.get("experiment"))
        print("  model_type:", ckpt.get("model_type"))
        print("  epoch:", ckpt.get("epoch"))

        model = build_fn(len(target_cols))
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        models.append(model)

    return models


@torch.no_grad()
def predict_batch(models, x, device):
    x = x.to(device, non_blocking=True)

    probs_sum = None

    for model in models:
        logits = model(x)
        probs = torch.sigmoid(logits)

        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum += probs

    return probs_sum / len(models)


def main():
    sample = pd.read_csv(DATA / "sample_submission.csv")
    target_cols = sample.columns[1:].astype(str).tolist()
    target_to_idx = {c: i for i, c in enumerate(target_cols)}

    df = pd.read_csv(XC_META)
    df["target_label"] = df["target_label"].astype(str)

    # Keep only rows whose target label is a competition target.
    df = df[df["target_label"].isin(target_to_idx)].reset_index(drop=True)

    print("xc candidate rows:", len(df))
    print("xc species:", df["target_label"].nunique())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    sed_models = load_models(SED_CKPT_PATHS, build_sed_model, target_cols, device, "SED")
    clip_models = load_models(CLIP_CKPT_PATHS, build_clip_model, target_cols, device, "clip")

    ds = XCMelDataset(df)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    all_probs = []
    all_indices = []

    for x, idx in tqdm(loader, desc="teacher filtering XC"):
        sed_probs = predict_batch(sed_models, x, device)
        clip_probs = predict_batch(clip_models, x, device)

        probs = SED_WEIGHT * sed_probs + CLIP_WEIGHT * clip_probs

        all_probs.append(probs.cpu().numpy().astype(np.float32))
        all_indices.append(idx.numpy())

    probs = np.concatenate(all_probs, axis=0)
    indices = np.concatenate(all_indices, axis=0)

    # Reorder defensively.
    order = np.argsort(indices)
    probs = probs[order]
    df = df.iloc[indices[order]].reset_index(drop=True)

    target_indices = df["target_label"].map(target_to_idx).values
    target_scores = probs[np.arange(len(df)), target_indices]

    df["teacher_target_score"] = target_scores

    keep = target_scores >= TARGET_THRESHOLD
    kept_df = df[keep].reset_index(drop=True)
    kept_probs = probs[keep].astype(np.float32)

    print()
    print("threshold:", TARGET_THRESHOLD)
    print("kept rows:", len(kept_df), "/", len(df))
    print("kept species:", kept_df["target_label"].nunique())
    print("score stats all:")
    print(pd.Series(target_scores).describe())
    print("score stats kept:")
    print(pd.Series(kept_df["teacher_target_score"]).describe() if len(kept_df) else "none")
    print("kept per species:")
    print(kept_df["target_label"].value_counts().head(50))

    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    kept_df.to_csv(OUT_META, index=False)
    np.save(OUT_PROBS, kept_probs)

    print("saved:", OUT_META)
    print("saved:", OUT_PROBS)
    print("probs shape:", kept_probs.shape)


if __name__ == "__main__":
    main()
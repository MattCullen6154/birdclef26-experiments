from pathlib import Path
import re

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
import timm


DATA = Path("data")
CKPT_PATH = Path("outputs/exp006_full_soundscapes/epoch06.pt")
OUT_PATH = Path("outputs/exp006_full_soundscapes/submission_local.csv")

SR = 32000
DURATION = 5
TARGET_LENGTH = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 16000

BATCH_SIZE = 64


def build_model(num_classes):
    model = timm.create_model(
        "tf_efficientnet_b0_ns",
        pretrained=False,
        in_chans=1,
        num_classes=num_classes,
    )
    return model


def normalize_audio(y):
    y = y.astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-6
    return y / peak


def audio_to_logmel_5s(y):
    if len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)))
    else:
        y = y[:TARGET_LENGTH]

    y = normalize_audio(y)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

    return logmel.astype(np.float32)


def parse_row_id(row_id):
    """
    Example row_id:
      BC2026_Test_0001_S05_20250227_010002_5

    We need:
      filename stem: BC2026_Test_0001_S05_20250227_010002
      end second: 5
      start second: 0
    """
    row_id = str(row_id)
    m = re.match(r"(.+)_(\d+)$", row_id)
    if m is None:
        raise ValueError(f"Could not parse row_id: {row_id}")

    stem = m.group(1)
    end_sec = int(m.group(2))
    start_sec = end_sec - DURATION

    filename = stem + ".ogg"
    return filename, start_sec, end_sec


@torch.no_grad()
def predict_batch(model, batch, device):
    x = torch.tensor(np.stack(batch), dtype=torch.float32).unsqueeze(1)
    x = x.to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)

    return probs.cpu().numpy()


def main():
    sample = pd.read_csv(DATA / "sample_submission.csv")
    target_cols = sample.columns[1:].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(CKPT_PATH, map_location=device)

    ckpt_targets = ckpt["target_cols"]
    if ckpt_targets != target_cols:
        raise ValueError("Checkpoint target columns do not match sample_submission columns")

    model = build_model(num_classes=len(target_cols))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    submission = sample.copy()

    # Cache loaded audio by filename since multiple rows usually come from same file.
    audio_cache = {}

    batch = []
    batch_indices = []

    pred_array = np.zeros((len(submission), len(target_cols)), dtype=np.float32)

    for idx, row in tqdm(submission.iterrows(), total=len(submission), desc="inference"):
        row_id = row["row_id"]
        filename, start_sec, end_sec = parse_row_id(row_id)

        audio_path = DATA / "test_soundscapes" / filename

        if filename not in audio_cache:
            y, _ = librosa.load(audio_path, sr=SR, mono=True)
            audio_cache[filename] = y
        else:
            y = audio_cache[filename]

        start_sample = int(start_sec * SR)
        end_sample = int(end_sec * SR)

        segment = y[start_sample:end_sample]
        logmel = audio_to_logmel_5s(segment)

        batch.append(logmel)
        batch_indices.append(idx)

        if len(batch) >= BATCH_SIZE:
            probs = predict_batch(model, batch, device)
            pred_array[batch_indices, :] = probs
            batch = []
            batch_indices = []

    if batch:
        probs = predict_batch(model, batch, device)
        pred_array[batch_indices, :] = probs

    submission[target_cols] = pred_array

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_PATH, index=False)

    print("saved:", OUT_PATH)
    print("shape:", submission.shape)
    print(submission.head())


if __name__ == "__main__":
    main()
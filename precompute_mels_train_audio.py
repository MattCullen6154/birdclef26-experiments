from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


DATA = Path("data")
OUT = Path("mels/train_audio")

SR = 32000
DURATION = 5
TARGET_LENGTH = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 16000


def normalize_audio(y):
    y = y.astype(np.float32)

    # Peak normalize to reduce source-volume variation.
    peak = np.max(np.abs(y)) + 1e-6
    y = y / peak

    return y


def choose_crop(y):
    """
    For precompute v1, use a center 5-second crop.
    Later we can make multiple crops per file or energy-based crops.
    """
    if len(y) < TARGET_LENGTH:
        y = np.pad(y, (0, TARGET_LENGTH - len(y)))
    else:
        start = max(0, (len(y) - TARGET_LENGTH) // 2)
        y = y[start:start + TARGET_LENGTH]

    return y


def make_logmel(audio_path):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    y = normalize_audio(y)
    y = choose_crop(y)

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

    # Per-sample normalization.
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

    return logmel.astype(np.float32)


def filename_to_npy_path(filename):
    """
    Example:
      train_audio/rubthr1/XC123.ogg
    becomes:
      mels/train_audio/rubthr1/XC123.npy
    """
    filename = str(filename).strip()
    rel = Path(filename)
    return OUT / rel.with_suffix(".npy")


def main():
    train_df = pd.read_csv(DATA / "train.csv")
    train_df["filename"] = train_df["filename"].astype(str).str.strip()

    OUT.mkdir(parents=True, exist_ok=True)

    failures = []
    skipped = 0
    written = 0

    for filename in tqdm(train_df["filename"].tolist(), desc="precomputing mels"):
        audio_path = DATA / "train_audio" / filename
        npy_path = filename_to_npy_path(filename)

        if npy_path.exists():
            skipped += 1
            continue

        npy_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logmel = make_logmel(audio_path)
            np.save(npy_path, logmel)
            written += 1
        except Exception as e:
            failures.append((filename, repr(e)))
            print(f"\nFAILED: {filename}")
            traceback.print_exc()

    print("\nDone.")
    print("written:", written)
    print("skipped:", skipped)
    print("failures:", len(failures))

    if failures:
        fail_path = OUT.parent / "precompute_failures.csv"
        pd.DataFrame(failures, columns=["filename", "error"]).to_csv(fail_path, index=False)
        print("failure log:", fail_path)

    # Quick sanity check.
    example_files = list(OUT.rglob("*.npy"))[:5]
    print("\nExample mel files:")
    for p in example_files:
        arr = np.load(p)
        print(p, arr.shape, arr.dtype, arr.min(), arr.max())


if __name__ == "__main__":
    main()
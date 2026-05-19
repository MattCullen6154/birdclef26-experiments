from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


DATA = Path("data")
OUT = Path("mels/train_audio_multicrop")
META_OUT = Path("mels/train_audio_multicrop_metadata.csv")

SR = 32000
DURATION = 5
TARGET_LENGTH = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 16000

RARE_THRESHOLD = 100
RANDOM_SEED = 42


def normalize_audio(y):
    y = y.astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-6
    return y / peak


def crop_or_pad(y, start):
    if len(y) < TARGET_LENGTH:
        return np.pad(y, (0, TARGET_LENGTH - len(y)))

    start = int(max(0, min(start, len(y) - TARGET_LENGTH)))
    return y[start:start + TARGET_LENGTH]


def center_crop(y):
    if len(y) <= TARGET_LENGTH:
        return crop_or_pad(y, 0)

    start = (len(y) - TARGET_LENGTH) // 2
    return crop_or_pad(y, start)


def random_crop(y, rng):
    if len(y) <= TARGET_LENGTH:
        return crop_or_pad(y, 0)

    start = rng.integers(0, len(y) - TARGET_LENGTH + 1)
    return crop_or_pad(y, start)


def max_energy_crop(y):
    """
    Choose the 5-second window with highest RMS energy.

    Uses coarse windows for speed. This is not sample-perfect exhaustive search,
    but it is good enough for picking an active region.
    """
    if len(y) <= TARGET_LENGTH:
        return crop_or_pad(y, 0)

    hop = SR  # evaluate every 1 sec
    starts = np.arange(0, len(y) - TARGET_LENGTH + 1, hop)

    if len(starts) == 0:
        return center_crop(y)

    best_start = 0
    best_energy = -1.0

    for s in starts:
        seg = y[s:s + TARGET_LENGTH]
        energy = float(np.mean(seg ** 2))
        if energy > best_energy:
            best_energy = energy
            best_start = int(s)

    return crop_or_pad(y, best_start)


def make_logmel(y):
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


def npy_path_for(filename, crop_type):
    filename = str(filename).strip()
    rel = Path(filename)
    return OUT / rel.parent / f"{rel.stem}__{crop_type}.npy"


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    train_df = pd.read_csv(DATA / "train.csv")
    train_df["filename"] = train_df["filename"].astype(str).str.strip()
    train_df["primary_label"] = train_df["primary_label"].astype(str).str.strip()
    train_df["secondary_labels"] = train_df["secondary_labels"].fillna("[]")

    counts = train_df["primary_label"].value_counts()
    train_df["class_count"] = train_df["primary_label"].map(counts)
    train_df["is_rare"] = train_df["class_count"] < RARE_THRESHOLD

    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    written = 0
    skipped = 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="precompute multicrop"):
        filename = row["filename"]
        audio_path = DATA / "train_audio" / filename

        crop_types = ["center"]

        if bool(row["is_rare"]):
            crop_types += ["maxenergy", "rand0", "rand1"]

        try:
            y, _ = librosa.load(audio_path, sr=SR, mono=True)

            for crop_type in crop_types:
                npy_path = npy_path_for(filename, crop_type)
                npy_path.parent.mkdir(parents=True, exist_ok=True)

                out_row = row.to_dict()
                out_row["source"] = "train_audio"
                out_row["crop_type"] = crop_type
                out_row["mel_path"] = str(npy_path)

                if npy_path.exists():
                    skipped += 1
                    rows.append(out_row)
                    continue

                if crop_type == "center":
                    y_crop = center_crop(y)
                elif crop_type == "maxenergy":
                    y_crop = max_energy_crop(y)
                elif crop_type in {"rand0", "rand1"}:
                    y_crop = random_crop(y, rng)
                else:
                    raise ValueError(f"Unknown crop_type: {crop_type}")

                logmel = make_logmel(y_crop)
                np.save(npy_path, logmel)

                written += 1
                rows.append(out_row)

        except Exception as e:
            failures.append((filename, repr(e)))
            print(f"\nFAILED: {filename}")
            traceback.print_exc()

    meta = pd.DataFrame(rows)
    meta.to_csv(META_OUT, index=False)

    print("\nDone.")
    print("written:", written)
    print("skipped:", skipped)
    print("failures:", len(failures))
    print("metadata:", META_OUT)
    print("metadata shape:", meta.shape)
    print("\nCrop counts:")
    print(meta["crop_type"].value_counts())
    print("\nRare rows:")
    print(meta["is_rare"].value_counts())

    if failures:
        fail_path = OUT.parent / "train_audio_multicrop_failures.csv"
        pd.DataFrame(failures, columns=["filename", "error"]).to_csv(fail_path, index=False)
        print("failure log:", fail_path)

    example_files = list(OUT.rglob("*.npy"))[:5]
    print("\nExample mel files:")
    for p in example_files:
        arr = np.load(p)
        print(p, arr.shape, arr.dtype, arr.min(), arr.max())


if __name__ == "__main__":
    main()
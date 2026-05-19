from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


DATA = Path("data")
OUT = Path("mels/train_soundscapes")
META_OUT = Path("mels/train_soundscapes_metadata.csv")

SR = 32000
DURATION = 5
TARGET_LENGTH = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 16000


def time_to_seconds(t):
    """
    Converts strings like:
      00:00:05
      00:01:10
    to seconds.
    """
    parts = str(t).strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad time format: {t}")

    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + int(s)


def normalize_audio(y):
    y = y.astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-6
    return y / peak


def make_logmel_from_segment(audio_path, start_sec, end_sec):
    offset = float(start_sec)
    duration = float(end_sec - start_sec)

    y, _ = librosa.load(
        audio_path,
        sr=SR,
        mono=True,
        offset=offset,
        duration=duration,
    )

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


def make_npy_path(filename, start_sec, end_sec):
    stem = Path(filename).stem
    return OUT / f"{stem}_{start_sec:05d}_{end_sec:05d}.npy"


def main():
    labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")

    labels["filename"] = labels["filename"].astype(str).str.strip()
    labels["start_sec"] = labels["start"].apply(time_to_seconds)
    labels["end_sec"] = labels["end"].apply(time_to_seconds)

    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    written = 0
    skipped = 0

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="precomputing soundscape mels"):
        filename = row["filename"]
        start_sec = int(row["start_sec"])
        end_sec = int(row["end_sec"])

        audio_path = DATA / "train_soundscapes" / filename
        npy_path = make_npy_path(filename, start_sec, end_sec)

        out_row = {
            "source": "soundscape",
            "filename": filename,
            "start": row["start"],
            "end": row["end"],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "primary_label": str(row["primary_label"]).strip(),
            "mel_path": str(npy_path),
        }

        if npy_path.exists():
            skipped += 1
            rows.append(out_row)
            continue

        try:
            logmel = make_logmel_from_segment(audio_path, start_sec, end_sec)
            np.save(npy_path, logmel)
            written += 1
            rows.append(out_row)
        except Exception as e:
            failures.append((filename, start_sec, end_sec, repr(e)))
            print(f"\nFAILED: {filename} {start_sec}-{end_sec}")
            traceback.print_exc()

    meta = pd.DataFrame(rows)
    meta.to_csv(META_OUT, index=False)

    print("\nDone.")
    print("written:", written)
    print("skipped:", skipped)
    print("failures:", len(failures))
    print("metadata:", META_OUT)

    if failures:
        fail_path = OUT.parent / "train_soundscapes_precompute_failures.csv"
        pd.DataFrame(
            failures,
            columns=["filename", "start_sec", "end_sec", "error"],
        ).to_csv(fail_path, index=False)
        print("failure log:", fail_path)

    example_files = list(OUT.rglob("*.npy"))[:5]
    print("\nExample mel files:")
    for p in example_files:
        arr = np.load(p)
        print(p, arr.shape, arr.dtype, arr.min(), arr.max())


if __name__ == "__main__":
    main()
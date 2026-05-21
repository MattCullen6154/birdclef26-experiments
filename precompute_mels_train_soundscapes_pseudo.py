from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


DATA = Path("data")
OUT = Path("mels/train_soundscapes_pseudo")
META_OUT = Path("mels/train_soundscapes_pseudo_metadata.csv")

SR = 32000
DURATION = 5
TARGET_LENGTH = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 16000

SEED = 42
MAX_PSEUDO_WINDOWS = 10000


def time_to_seconds(t):
    h, m, s = str(t).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds_to_hhmmss(sec):
    sec = int(sec)
    h = sec // 3600
    sec %= 3600
    m = sec // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_audio_duration_sec(path):
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def normalize_audio(y):
    y = y.astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-6
    return y / peak


def make_logmel_from_segment(audio_path, start_sec, end_sec):
    y, _ = librosa.load(
        audio_path,
        sr=SR,
        mono=True,
        offset=float(start_sec),
        duration=float(end_sec - start_sec),
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
    return OUT / f"{stem}_{start_sec:05d}_{end_sec:05d}_pseudo.npy"


def main():
    rng = np.random.default_rng(SEED)

    labels = pd.read_csv(DATA / "train_soundscapes_labels.csv")
    labels["filename"] = labels["filename"].astype(str).str.strip()
    labels["start_sec"] = labels["start"].apply(time_to_seconds)
    labels["end_sec"] = labels["end"].apply(time_to_seconds)

    labeled_intervals = set(
        zip(
            labels["filename"],
            labels["start_sec"].astype(int),
            labels["end_sec"].astype(int),
        )
    )

    soundscape_files = sorted((DATA / "train_soundscapes").glob("*.ogg"))
    print("train_soundscape files:", len(soundscape_files))
    print("official labeled intervals:", len(labeled_intervals))

    candidates = []

    for audio_path in tqdm(soundscape_files, desc="building pseudo candidates"):
        filename = audio_path.name
        duration = get_audio_duration_sec(audio_path)
        max_end = int(duration // DURATION) * DURATION

        for start_sec in range(0, max_end, DURATION):
            end_sec = start_sec + DURATION

            if (filename, start_sec, end_sec) in labeled_intervals:
                continue

            candidates.append(
                {
                    "source": "pseudo",
                    "filename": filename,
                    "start": seconds_to_hhmmss(start_sec),
                    "end": seconds_to_hhmmss(end_sec),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "primary_label": "",
                    "secondary_labels": "[]",
                }
            )

    print("unlabeled candidate windows:", len(candidates))

    rng.shuffle(candidates)
    selected = candidates[: min(MAX_PSEUDO_WINDOWS, len(candidates))]
    print("selected pseudo windows:", len(selected))

    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    written = 0
    skipped = 0

    for pseudo_idx, row in enumerate(tqdm(selected, desc="precomputing pseudo mels")):
        filename = row["filename"]
        start_sec = int(row["start_sec"])
        end_sec = int(row["end_sec"])

        audio_path = DATA / "train_soundscapes" / filename
        npy_path = make_npy_path(filename, start_sec, end_sec)

        out_row = dict(row)
        out_row["mel_path"] = str(npy_path)
        out_row["pseudo_idx"] = pseudo_idx

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
    print("metadata shape:", meta.shape)

    if failures:
        fail_path = OUT.parent / "train_soundscapes_pseudo_failures.csv"
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
    
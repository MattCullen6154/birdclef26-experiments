from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


XC_META = Path("external/xenocanto/xenocanto_downloads.csv")
OUT_DIR = Path("mels/xenocanto_pseudo")
OUT_META = Path("mels/xenocanto_pseudo_metadata.csv")

SR = 32000
DURATION = 5
TARGET_LENGTH = SR * DURATION

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 16000

MAX_WINDOWS_PER_RECORDING = 8


def normalize_audio(y):
    y = y.astype(np.float32)
    peak = np.max(np.abs(y)) + 1e-6
    return y / peak


def make_logmel(y):
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


def main():
    meta = pd.read_csv(XC_META)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, row in tqdm(meta.iterrows(), total=len(meta), desc="xc recordings"):
        path = Path(row["local_audio_path"])
        if not path.exists():
            continue

        try:
            y_full, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print("SKIP bad audio file:", path, repr(e))
            continue

        duration = len(y_full) / SR
        n_windows = int(duration // DURATION)
        n_windows = min(n_windows, MAX_WINDOWS_PER_RECORDING)

        if n_windows <= 0:
            continue

        for w in range(n_windows):
            start_sample = w * TARGET_LENGTH
            end_sample = start_sample + TARGET_LENGTH

            y = y_full[start_sample:end_sample]

            if len(y) < TARGET_LENGTH:
                continue

            try:
                logmel = make_logmel(y)

                target_label = str(row["target_label"])
                xc_id = str(row["id"])
                out_path = OUT_DIR / target_label / f"XC{xc_id}_{w:03d}.npy"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                np.save(out_path, logmel)

                rows.append(
                    {
                        "source": "xenocanto",
                        "target_label": target_label,
                        "primary_label": target_label,
                        "secondary_labels": "[]",
                        "xc_id": xc_id,
                        "filename": path.name,
                        "local_audio_path": str(path),
                        "start_sec": w * DURATION,
                        "end_sec": (w + 1) * DURATION,
                        "mel_path": str(out_path),
                        "recordist": row.get("rec", ""),
                        "quality": row.get("q", ""),
                    }
                )

            except Exception as e:
                print("FAILED window:", path, w, repr(e))
                continue

    out = pd.DataFrame(rows)
    out.to_csv(OUT_META, index=False)

    print("windows:", len(out))
    print("saved:", OUT_META)
    print(out.head().to_string())


if __name__ == "__main__":
    main()
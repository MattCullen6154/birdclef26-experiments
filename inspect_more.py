from pathlib import Path
import pandas as pd

DATA = Path("data")

train = pd.read_csv(DATA / "train.csv")
tax = pd.read_csv(DATA / "taxonomy.csv")
sample = pd.read_csv(DATA / "sample_submission.csv")

print("Train shape:", train.shape)
print("Taxonomy shape:", tax.shape)
print("Sample submission shape:", sample.shape)

print("\nTarget columns:")
target_cols = sample.columns[1:].tolist()
print("num targets:", len(target_cols))
print(target_cols[:10])

print("\nDo sample columns match taxonomy labels?")
taxonomy_labels = tax["primary_label"].astype(str).tolist()
print(set(target_cols) == set(taxonomy_labels))

print("\nClass distribution:")
print(train["primary_label"].value_counts().describe())
print(train["primary_label"].value_counts().head(20))
print(train["primary_label"].value_counts().tail(20))

print("\nClass names:")
print(train["class_name"].value_counts())

print("\nRatings:")
print(train["rating"].describe())
print(train["rating"].value_counts(dropna=False).sort_index())

print("\nExample filenames:")
print(train["filename"].head(20).to_string(index=False))

print("\nAudio files count:")
audio_files = list((DATA / "train_audio").rglob("*.ogg"))
print(len(audio_files))
print(audio_files[:5])

print("\nTrain soundscape labels:")
ssl_path = DATA / "train_soundscapes_labels.csv"
if ssl_path.exists():
    ssl = pd.read_csv(ssl_path)
    print(ssl.shape)
    print(ssl.head())
    print(ssl.columns.tolist())

print("\nTrain soundscapes:")
soundscape_files = list((DATA / "train_soundscapes").rglob("*"))
soundscape_files = [p for p in soundscape_files if p.is_file()]
print(len(soundscape_files))
print(soundscape_files[:5])
from pathlib import Path
import pandas as pd
DATA = Path("data")

print("Top-level files/folders:")
for p in DATA.iterdir():
    print(" ", p)

print("\nCSV summaries:")
for csv_path in sorted(DATA.glob("*.csv")):
    print(f"\n=== {csv_path.name} ===")
    df = pd.read_csv(csv_path)
    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print(df.head())
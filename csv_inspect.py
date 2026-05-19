import pandas as pd
from pathlib import Path

DATA = Path("data")

for p in DATA.glob("*.csv"):
    print("\n", p.name)
    df = pd.read_csv(p)
    print(df.shape)
    print(df.head())
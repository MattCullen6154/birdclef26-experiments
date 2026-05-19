from pathlib import Path
import pandas as pd

ROOT = Path("outputs/exp005_multisplit")

histories = []
for hist_path in ROOT.glob("seed_*/history.csv"):
    df = pd.read_csv(hist_path)
    histories.append(df)

all_hist = pd.concat(histories, ignore_index=True)

summary = (
    all_hist
    .groupby("epoch")
    .agg(
        mean_auc=("val_auc", "mean"),
        std_auc=("val_auc", "std"),
        min_auc=("val_auc", "min"),
        max_auc=("val_auc", "max"),
        mean_loss=("val_loss", "mean"),
        mean_scored_classes=("scored_classes", "mean"),
    )
    .reset_index()
)

print(summary.to_string(index=False))

best = summary.sort_values("mean_auc", ascending=False).iloc[0]
print("\nBest mean-AUC epoch:")
print(best)

summary.to_csv(ROOT / "mean_by_epoch.csv", index=False)
print("\nsaved:", ROOT / "mean_by_epoch.csv")

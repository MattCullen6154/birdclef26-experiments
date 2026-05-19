import pandas as pd

sample = pd.read_csv("data/sample_submission.csv")
submission = sample.copy()

# tiny nonzero constant predictions
for col in submission.columns:
    if col != "row_id":
        submission[col] = 0.001

submission.to_csv("submission.csv", index=False)
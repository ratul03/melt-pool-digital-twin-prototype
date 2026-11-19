from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
d = pd.read_csv(root / "results/primary_per_alloy_per_seed.csv")
metrics = ["R2", "MAE", "RMSE", "accuracy", "macro_f1", "balanced_accuracy"]
# Each eligible alloy receives equal weight within each seed, unlike pooled scores.
s = d.groupby(["task", "protocol", "seed"])[metrics].mean()
s.to_csv(root / "results/equal_alloy_per_seed.csv")
s.groupby(["task", "protocol"]).agg(["mean", "std", "count"]).to_csv(
    root / "results/equal_alloy_summary.csv"
)
print("Saved equal-alloy summaries. These average within-alloy metrics, not pooled predictions.")

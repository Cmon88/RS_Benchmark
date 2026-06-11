import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path

BASE = Path(__file__).parent.parent

ablation = pd.read_csv(BASE / "structural_perturbation" / "ablation_results_100k.csv")
agg = pd.read_csv(BASE / "results_100k" / "aggregated_best_metrics.csv")

pattern = re.compile(r"^(.+)_sample(\d+)_train$")
parsed = ablation["dataset_name"].str.extract(pattern)
ablation["dataset"] = parsed[0]
ablation["sample"] = parsed[1].astype(int)
ablation = ablation.rename(columns={
    "Structural Perturbation RMSE": "SC_RMSE",
    "Spectral Distance": "Spectral_Distance",
})

agg = agg.rename(columns={"Sample": "sample"})
agg["sample"] = agg["sample"].astype(int)
rec_cols = ["mrr@10", "ndcg@10", "precision@10", "recall@10"]
agg = agg[["dataset", "sample"] + rec_cols]

merged = ablation.merge(agg, on=["dataset", "sample"], how="inner")
print(f"Merged shape: {merged.shape}  (expect ~1275 rows)")
print(f"Datasets in merged ({merged['dataset'].nunique()}):")
for ds in sorted(merged['dataset'].unique()):
    print(f"  {ds}")

p_vals = sorted(merged["p"].unique())
alpha_vals = sorted(merged["alpha"].unique())
struct_measures = ["SC_RMSE", "Spectral_Distance"]
struct_labels = ["SC RMSE", "Spectral Distance"]

spearman_grids = {}
for sm in struct_measures:
    for rc in rec_cols:
        grid = np.full((len(p_vals), len(alpha_vals)), np.nan)
        for i, p in enumerate(p_vals):
            for j, a in enumerate(alpha_vals):
                subset = merged[(merged["p"] == p) & (merged["alpha"] == a)]
                if len(subset) >= 3:
                    grid[i, j] = spearmanr(subset[sm], subset[rc]).statistic
        spearman_grids[(sm, rc)] = grid
        print(f"{sm} vs {rc}:\n{np.round(grid, 3)}\n")

rec_labels = ["MRR@10", "NDCG@10", "Precision@10", "Recall@10"]
out_dir = BASE / "structural_perturbation"

for sm, sm_label in zip(struct_measures, struct_labels):
    for rc, rc_label in zip(rec_cols, rec_labels):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            spearman_grids[(sm, rc)],
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            xticklabels=alpha_vals,
            yticklabels=p_vals,
            annot_kws={"size": 16, "color": "black"},
        )
        ax.set_xlabel("α", fontsize=16)
        ax.set_ylabel("p", fontsize=16)
        ax.tick_params(labelsize=15)
        slug = f"{sm_label.replace(' ', '_')}__vs__{rc_label.replace('@', '_at_')}"
        out_path = out_dir / f"spearman_{slug}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)

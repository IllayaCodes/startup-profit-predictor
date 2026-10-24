# %% [markdown]
# # Step 4: Model comparison with repeated cross-validation

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def find_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "50_Startups.csv").exists():
            return p
    raise FileNotFoundError("Open the project folder in VS Code and run again.")


ROOT = find_root()
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_SETS, TARGET, load_startups, make_features

# %%
df = load_startups(ROOT / "data" / "50_Startups.csv")
X_all = make_features(df.drop(columns=[TARGET]))   # profit is never given to the features
y = df[TARGET]
print(X_all.shape, y.shape)

# %%
models = {
    "Linear": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
}

# 5 folds, repeated 10 times = 50 scores per model, so we can see the spread
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scoring = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}

# %% [markdown]
# This next cell can take a minute or two.

# %%
rows = []
fold_rmse = {}

for fs_name, cols in FEATURE_SETS.items():
    X = X_all[cols]
    for m_name, model in models.items():
        res = cross_validate(clone(model), X, y, cv=cv, scoring=scoring)
        rmse = -res["test_rmse"]
        mae = -res["test_mae"]
        r2 = res["test_r2"]
        fold_rmse[(fs_name, m_name)] = rmse
        rows.append({
            "features": fs_name,
            "model": m_name,
            "r2_mean": r2.mean(),
            "r2_std": r2.std(),
            "rmse_mean": rmse.mean(),
            "rmse_std": rmse.std(),
            "mae_mean": mae.mean(),
        })

results = pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)

# baseline = plain linear model on R&D only
base = results[(results["features"] == "rd_only") & (results["model"] == "Linear")].iloc[0]
results["rmse_vs_baseline"] = results["rmse_mean"] - base["rmse_mean"]
results["clearly_better"] = results["rmse_vs_baseline"] < -base["rmse_std"]

print(f"Baseline (Linear, rd_only): RMSE {base['rmse_mean']:,.0f} +/- {base['rmse_std']:,.0f}, R2 {base['r2_mean']:.3f}")
display(results.round(3))

# %%
results.round(4).to_csv(ROOT / "docs" / "cv_results.csv", index=False)

top = results.head(8)
labels = [f"{r.features}\n{r.model}" for r in top.itertuples()]
data = [fold_rmse[(r.features, r.model)] for r in top.itertuples()]

plt.figure(figsize=(12, 5.5))
plt.boxplot(data, tick_labels=labels)
plt.axhline(base["rmse_mean"], color="crimson", linestyle="--", label="baseline mean RMSE")
plt.xticks(rotation=30, ha="right")
plt.ylabel("RMSE across 50 folds (lower is better)")
plt.title("Top 8 combinations")
plt.legend()
plt.tight_layout()
plt.show()
# %%

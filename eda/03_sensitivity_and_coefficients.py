# %% [markdown]
# # Step 5: Sensitivity check, coefficients and errors

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict, cross_validate


def find_root() -> Path:
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "50_Startups.csv").exists():
            return p
    raise FileNotFoundError("Open the project folder in VS Code and run again.")


ROOT = find_root()
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_SETS, SPEND_COLS, TARGET, load_startups, make_features

RNG = np.random.default_rng(42)

df = load_startups(ROOT / "data" / "50_Startups.csv")
X_all = make_features(df.drop(columns=[TARGET]))
y = df[TARGET]
zero_mask = (df[SPEND_COLS] == 0).any(axis=1)
print("Rows with a zero spend:", list(df.index[zero_mask]))

# %% [markdown]
# ## 5.1 Do the results change without the 4 zero-spend rows?

# %%
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
candidates = {
    "Linear, rd_only": (LinearRegression(), FEATURE_SETS["rd_only"]),
    "Linear, raw_spends": (LinearRegression(), FEATURE_SETS["raw_spends"]),
    "ExtraTrees, raw_spends": (ExtraTreesRegressor(n_estimators=200, random_state=42), FEATURE_SETS["raw_spends"]),
}
subsets = {
    "all 50 rows": pd.Series(True, index=df.index),
    "without the 4 zero-spend rows": ~zero_mask,
}

rows = []
for subset_name, keep in subsets.items():
    for label, (model, cols) in candidates.items():
        res = cross_validate(model, X_all.loc[keep, cols], y[keep], cv=cv,
                             scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"})
        rows.append({
            "rows used": subset_name,
            "model": label,
            "n": int(keep.sum()),
            "r2_mean": res["test_r2"].mean(),
            "rmse_mean": -res["test_rmse"].mean(),
            "rmse_std": res["test_rmse"].std(),
        })

display(pd.DataFrame(rows).round(3))

# %% [markdown]
# ## 5.2 Linear coefficients with bootstrap intervals
# Each coefficient is the change in profit (in dollars) linked to one extra dollar of that spend,
# holding the other two fixed. This is an association in 50 startups, not a cause.

# %%
cols = FEATURE_SETS["raw_spends"]
X, yy = X_all[cols], y
fit = LinearRegression().fit(X, yy)

boot = []
for _ in range(2000):
    idx = RNG.integers(0, len(X), len(X))
    m = LinearRegression().fit(X.iloc[idx], yy.iloc[idx])
    boot.append(m.coef_)
boot = np.array(boot)

coef = pd.DataFrame({
    "coef": fit.coef_,
    "ci95_low": np.percentile(boot, 2.5, axis=0),
    "ci95_high": np.percentile(boot, 97.5, axis=0),
}, index=cols)
display(coef.round(3))
print("Intercept:", round(fit.intercept_))

# %% [markdown]
# ## 5.3 Where does the model make its biggest mistakes?

# %%
pred = cross_val_predict(LinearRegression(), X_all[cols], y,
                         cv=KFold(5, shuffle=True, random_state=42))
resid = y - pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(pred, y, alpha=0.7)
lims = [y.min(), y.max()]
axes[0].plot(lims, lims, "r--")
axes[0].set_xlabel("predicted profit (cross-validated)")
axes[0].set_ylabel("actual profit")
axes[0].set_title("Actual vs predicted")

axes[1].scatter(pred, resid, alpha=0.7)
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_xlabel("predicted profit (cross-validated)")
axes[1].set_ylabel("error (actual - predicted)")
axes[1].set_title("Errors")
plt.tight_layout()
plt.show()

print("Largest errors (row number, size):")
display(resid.abs().sort_values(ascending=False).head(5).round(0).to_frame("abs_error"))
# %%


# %% [markdown]
# # Step 1: Setup and data audit

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
RNG = np.random.default_rng(42)


# %%
def find_root() -> Path:
    """Walk up from the current folder until we find data/50_Startups.csv."""
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "50_Startups.csv").exists():
            return p
    raise FileNotFoundError("Could not find data/50_Startups.csv. Open the project folder in VS Code.")


ROOT = find_root()
print("Project root:", ROOT)

# %%
df = pd.read_csv(ROOT / "data" / "50_Startups.csv")
df.columns = [c.strip() for c in df.columns]

df = df.rename(columns={
    "R&D Spend": "rd_spend",
    "Administration": "admin_spend",
    "Marketing Spend": "marketing_spend",
    "Profit": "profit",
})

SPEND = ["rd_spend", "admin_spend", "marketing_spend"]
TARGET = "profit"
display(df.head())

# %%
print("Shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

audit = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "nulls": df.isna().sum(),
    "zeros": (df == 0).sum(),
})
display(audit)
display(df.describe().T.round(2))

# %%
zero_rows = df[(df[SPEND] == 0).any(axis=1)]
print(f"Startups with at least one zero spend: {len(zero_rows)} of {len(df)}")
display(zero_rows)
# %%

# %% [markdown]
# ## Step 2.1: Distributions

# %%
cols = SPEND + [TARGET]
fig, axes = plt.subplots(2, len(cols), figsize=(4.5 * len(cols), 7))

for i, c in enumerate(cols):
    sns.histplot(df[c], kde=True, ax=axes[0, i], color="steelblue")
    axes[0, i].set_title(c)
    sns.boxplot(x=df[c], ax=axes[1, i], color="lightsteelblue")

plt.tight_layout()
plt.show()

print("Skewness:")
print(df[cols].skew().round(2))

# %% [markdown]
# ## Step 2.2: Each spend vs. profit

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

for ax, c in zip(axes, SPEND):
    sns.regplot(data=df, x=c, y=TARGET, ax=ax, ci=95,
                scatter_kws={"alpha": 0.7}, line_kws={"color": "crimson"})
    rho, p = stats.spearmanr(df[c], df[TARGET])
    ax.set_title(f"{c} (Spearman rho = {rho:.2f})")

plt.tight_layout()
plt.show()
# %%

# %% [markdown]
# ## Step 2.3: Correlation and multicollinearity

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, method in zip(axes, ["pearson", "spearman"]):
    sns.heatmap(df[SPEND + [TARGET]].corr(method=method), annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title(method.title())
plt.tight_layout()
plt.show()

# Variance Inflation Factor from the inputs' correlation matrix.
# Rule of thumb: above about 5 means a feature is largely explained by the others.
corr_inputs = df[SPEND].corr().to_numpy()
vif = pd.Series(np.diag(np.linalg.inv(corr_inputs)), index=SPEND, name="VIF").round(2)
display(vif.to_frame())

# %% [markdown]
# ## Step 2.4: Zero-spend startups vs. the rest, and outliers

# %%
zero_mask = (df[SPEND] == 0).any(axis=1)
print(f"Zero-spend rows: n = {zero_mask.sum()}, mean profit = {df.loc[zero_mask, TARGET].mean():,.0f}")
print(f"Other rows:      n = {(~zero_mask).sum()}, mean profit = {df.loc[~zero_mask, TARGET].mean():,.0f}")

cols = SPEND + [TARGET]
q1, q3 = df[cols].quantile(0.25), df[cols].quantile(0.75)
iqr = q3 - q1
outliers = (df[cols] < q1 - 1.5 * iqr) | (df[cols] > q3 + 1.5 * iqr)
print("\nIQR outliers per column:")
print(outliers.sum())
display(df[outliers.any(axis=1)])
# %%

# %% [markdown]
# ## Step 2.5: Budget mix, top 25% vs. the rest

# %%
total = df[SPEND].sum(axis=1)
mix = df[SPEND].div(total, axis=0).add_suffix("_share")   # spend-only, safe as model features later
mix[TARGET] = df[TARGET]

top_cut = df[TARGET].quantile(0.75)
mix["group"] = np.where(df[TARGET] > top_cut, "Top 25% profit", "Rest")

share_cols = [c + "_share" for c in SPEND]
summary = mix.groupby("group")[share_cols].mean().mul(100).round(1)
display(summary)

summary.T.plot(kind="bar", rot=0, figsize=(8, 4.5), title="Average budget share (%)")
plt.ylabel("% of total spend")
plt.show()

print("Does the share relate to profit?")
for c in share_cols:
    rho, p = stats.spearmanr(mix[c], mix[TARGET])
    print(f"{c:<24} rho = {rho:+.2f}, p = {p:.3f}")

print("\nMean total spend by group:")
display(total.groupby(mix["group"]).mean().round(0).to_frame("total_spend"))

# %% [markdown]
# ## Step 2.6: Hypothesis check with bootstrap intervals

# %%
def boot_spearman_ci(x, y, n_boot=2000):
    x, y = x.to_numpy(), y.to_numpy()
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    rs = [stats.spearmanr(x[i], y[i])[0] for i in idx]
    return np.nanpercentile(rs, [2.5, 97.5])


rows = []
for c in SPEND:
    rho, p = stats.spearmanr(df[c], df[TARGET])
    lo, hi = boot_spearman_ci(df[c], df[TARGET])
    rows.append({"feature": c, "spearman_rho": round(rho, 3), "p_value": round(p, 4),
                 "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)})

hyp = pd.DataFrame(rows).set_index("feature")
display(hyp)
# %%

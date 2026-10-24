
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
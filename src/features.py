"""Feature code shared by training and the Streamlit app.

Rule: nothing in this file may read the profit column.
A feature is allowed only if a user could enter its inputs
without knowing the profit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SPEND_COLS = ["rd_spend", "admin_spend", "marketing_spend"]
TARGET = "profit"

RENAME = {
    "R&D Spend": "rd_spend",
    "Administration": "admin_spend",
    "Marketing Spend": "marketing_spend",
    "Profit": "profit",
}


def load_startups(path) -> pd.DataFrame:
    """Read the CSV and give the columns short names."""
    df = pd.read_csv(Path(path))
    df.columns = [c.strip() for c in df.columns]
    return df.rename(columns=RENAME)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build every feature from the three spend columns only."""
    missing = [c for c in SPEND_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing spend columns: {missing}")

    out = df[SPEND_COLS].astype(float).copy()

    total = out.sum(axis=1)
    out["total_spend"] = total

    # if total is 0 the shares are undefined, so use 0 instead of NaN
    safe_total = total.replace(0, np.nan)
    out["rd_share"] = (out["rd_spend"] / safe_total).fillna(0.0)
    out["admin_share"] = (out["admin_spend"] / safe_total).fillna(0.0)
    out["marketing_share"] = (out["marketing_spend"] / safe_total).fillna(0.0)

    # log1p keeps zero-spend startups working (plain log would break on 0)
    out["rd_marketing_log_ratio"] = np.log1p(out["rd_spend"]) - np.log1p(out["marketing_spend"])

    return out


# Feature sets we will compare in the modelling step.
# The three shares always add up to 1, and total_spend is the sum of the raw
# spends, so a linear model must not get all of those together.
FEATURE_SETS = {
    "rd_only": ["rd_spend"],
    "raw_spends": ["rd_spend", "admin_spend", "marketing_spend"],
    "shares_and_total": ["rd_share", "marketing_share", "total_spend"],
    "all_engineered": [
        "rd_spend", "admin_spend", "marketing_spend", "total_spend",
        "rd_share", "admin_share", "marketing_share", "rd_marketing_log_ratio",
    ],
}
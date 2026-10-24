"""Train, save, load and use the final profit model.

The app imports predict_profit from here, so training and the app always
build their features with the same function in src/features.py.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold

from src.features import FEATURE_SETS, SPEND_COLS, TARGET, load_startups, make_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "50_Startups.csv"
MODEL_PATH = ROOT / "models" / "startup_profit_model.joblib"

FEATURES = FEATURE_SETS["raw_spends"]   # the three raw spends
LEVEL = 0.90                            # width of the prediction range


def train_and_save(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = load_startups(data_path)
    X = make_features(df.drop(columns=[TARGET]))[FEATURES]
    y = df[TARGET]

    # Out-of-fold errors: every error comes from a model that did not see that startup.
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    row_ids, resid = [], []
    for train_idx, test_idx in cv.split(X):
        m = LinearRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
        err = y.iloc[test_idx].to_numpy() - m.predict(X.iloc[test_idx])
        row_ids.extend(test_idx)
        resid.extend(err)
    row_ids, resid = np.array(row_ids), np.array(resid)

    lo_p, hi_p = (1 - LEVEL) / 2 * 100, (1 + LEVEL) / 2 * 100
    low_q, high_q = np.percentile(resid, [lo_p, hi_p])

    # Honest check: for each startup, build the range from the OTHER startups' errors
    # and see how often that startup's own errors land inside it.
    inside = []
    for i in range(len(X)):
        lo, hi = np.percentile(resid[row_ids != i], [lo_p, hi_p])
        mine = resid[row_ids == i]
        inside.append(((mine >= lo) & (mine <= hi)).mean())

    model = LinearRegression().fit(X, y)

    bundle = {
        "model": model,
        "features": FEATURES,
        "low_q": float(low_q),
        "high_q": float(high_q),
        "level": LEVEL,
        "coverage_check": float(np.mean(inside)),
        "cv_rmse": float(np.sqrt(np.mean(resid ** 2))),
        "cv_mae": float(np.mean(np.abs(resid))),
        "train_min": df[SPEND_COLS].min().to_dict(),
        "train_max": df[SPEND_COLS].max().to_dict(),
        "n_train": int(len(df)),
        "coefficients": dict(zip(FEATURES, model.coef_.tolist())),
        "intercept": float(model.intercept_),
        "sklearn_version": sklearn.__version__,
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def load_bundle(model_path=MODEL_PATH):
    return joblib.load(model_path)


def predict_profit(bundle, rd_spend, admin_spend, marketing_spend):
    """Return the predicted profit, a range, and any warnings about the inputs."""
    values = {"rd_spend": rd_spend, "admin_spend": admin_spend, "marketing_spend": marketing_spend}
    if any(v < 0 for v in values.values()):
        raise ValueError("Spend values cannot be negative.")

    X = make_features(pd.DataFrame([values]))[bundle["features"]]
    pred = float(bundle["model"].predict(X)[0])

    warnings = []
    for col, val in values.items():
        lo, hi = bundle["train_min"][col], bundle["train_max"][col]
        if val < lo or val > hi:
            warnings.append(
                f"{col} = {val:,.0f} is outside the range the model learned from "
                f"({lo:,.0f} to {hi:,.0f}), so treat this result with care."
            )

    return {
        "prediction": pred,
        "low": pred + bundle["low_q"],
        "high": pred + bundle["high_q"],
        "level": bundle["level"],
        "warnings": warnings,
    }


if __name__ == "__main__":
    b = train_and_save()
    print(f"Trained on {b['n_train']} startups, saved to {MODEL_PATH}")
    print(f"Cross-validated RMSE: {b['cv_rmse']:,.0f}   MAE: {b['cv_mae']:,.0f}")
    print(f"{int(b['level'] * 100)}% range: prediction {b['low_q']:+,.0f} to {b['high_q']:+,.0f}")
    print(f"Coverage check (range built without the startup it is tested on): {b['coverage_check']:.0%}")
    print("Coefficients:", {k: round(v, 3) for k, v in b["coefficients"].items()}, "| intercept:", round(b["intercept"]))

    for label, spends in {
        "typical startup": (73_000, 121_000, 211_000),
        "R&D far above training range": (500_000, 121_000, 211_000),
    }.items():
        r = predict_profit(b, *spends)
        print(f"\n{label}: {r['prediction']:,.0f}  (range {r['low']:,.0f} to {r['high']:,.0f})")
        for w in r["warnings"]:
            print("  WARNING:", w)
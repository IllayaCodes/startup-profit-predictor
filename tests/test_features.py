import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_SETS, SPEND_COLS, make_features


def sample_df(profit=None):
    df = pd.DataFrame({
        "rd_spend": [100_000.0, 0.0, 50_000.0],
        "admin_spend": [120_000.0, 90_000.0, 0.0],
        "marketing_spend": [200_000.0, 0.0, 40_000.0],
    })
    if profit is not None:
        df["profit"] = profit
    return df


def test_features_work_without_profit_column():
    X = make_features(sample_df())
    assert len(X) == 3


def test_features_do_not_change_when_profit_changes():
    a = make_features(sample_df(profit=[1.0, 2.0, 3.0]))
    b = make_features(sample_df(profit=[999.0, -5.0, 1e9]))
    pd.testing.assert_frame_equal(a, b)


def test_no_feature_set_uses_profit():
    for cols in FEATURE_SETS.values():
        assert "profit" not in cols


def test_shares_add_up_to_one():
    X = make_features(sample_df())
    total_share = X["rd_share"] + X["admin_share"] + X["marketing_share"]
    assert np.allclose(total_share, 1.0)


def test_zero_total_spend_gives_zero_shares_not_nan():
    df = pd.DataFrame({"rd_spend": [0.0], "admin_spend": [0.0], "marketing_spend": [0.0]})
    X = make_features(df)
    assert X.isna().sum().sum() == 0
    assert X.loc[0, "rd_share"] == 0.0


def test_zero_marketing_does_not_break_the_log_ratio():
    X = make_features(sample_df())
    assert np.isfinite(X["rd_marketing_log_ratio"]).all()


def test_missing_spend_column_raises():
    with pytest.raises(ValueError):
        make_features(sample_df().drop(columns=["marketing_spend"]))
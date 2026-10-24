import pytest

from src.features import FEATURE_SETS, TARGET, load_startups, make_features
from src.model import DATA_PATH, MODEL_PATH, load_bundle, predict_profit


@pytest.fixture(scope="module")
def bundle():
    if not MODEL_PATH.exists():
        pytest.skip("Run python -m src.model first to create the saved model.")
    return load_bundle(MODEL_PATH)


def test_range_surrounds_prediction(bundle):
    r = predict_profit(bundle, 73_000, 121_000, 211_000)
    assert r["low"] < r["prediction"] < r["high"]


def test_typical_inputs_have_no_warning(bundle):
    r = predict_profit(bundle, 73_000, 121_000, 211_000)
    assert r["warnings"] == []


def test_input_above_training_range_warns(bundle):
    r = predict_profit(bundle, 500_000, 121_000, 211_000)
    assert len(r["warnings"]) == 1


def test_negative_spend_is_rejected(bundle):
    with pytest.raises(ValueError):
        predict_profit(bundle, -1, 121_000, 211_000)


def test_prediction_matches_the_saved_coefficients(bundle):
    rd, admin, mkt = 80_000, 110_000, 150_000
    c = bundle["coefficients"]
    expected = (bundle["intercept"] + c["rd_spend"] * rd
                + c["admin_spend"] * admin + c["marketing_spend"] * mkt)
    assert predict_profit(bundle, rd, admin, mkt)["prediction"] == pytest.approx(expected)


def test_app_function_matches_training_features(bundle):
    """The app path and the training path must give the same numbers."""
    df = load_startups(DATA_PATH)
    X = make_features(df.drop(columns=[TARGET]))[FEATURE_SETS["raw_spends"]]
    direct = bundle["model"].predict(X.head(5))
    for i in range(5):
        row = df.iloc[i]
        via_app = predict_profit(bundle, row["rd_spend"], row["admin_spend"], row["marketing_spend"])
        assert via_app["prediction"] == pytest.approx(direct[i])


def test_coverage_check_is_reasonable(bundle):
    assert 0.80 <= bundle["coverage_check"] <= 0.99
"""Startup Profit Predictor (Streamlit app)"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import SPEND_COLS, TARGET, load_startups  # noqa: E402
from src.model import DATA_PATH, MODEL_PATH, load_bundle, predict_profit  # noqa: E402

st.set_page_config(page_title="Startup Profit Predictor", layout="wide")


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


@st.cache_data
def get_data():
    return load_startups(DATA_PATH)


bundle = get_bundle()
df = get_data()

LABELS = {
    "rd_spend": "R&D spend",
    "admin_spend": "Administration spend",
    "marketing_spend": "Marketing spend",
}

st.title("Startup Profit Predictor")
st.caption(
    "A learning project built on 50 startups. It shows what a budget split goes with "
    "in this data, not what will happen to a real company."
)

st.sidebar.header("Your budget")
inputs = {}
for col in SPEND_COLS:
    default = float(round(df[col].median(), -3))
    inputs[col] = st.sidebar.number_input(
        LABELS[col], min_value=0.0, value=default, step=1000.0, format="%.0f"
    )

result = predict_profit(
    bundle, inputs["rd_spend"], inputs["admin_spend"], inputs["marketing_spend"]
)
pred, low, high = result["prediction"], result["low"], result["high"]
level = int(result["level"] * 100)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted profit", f"{pred:,.0f}")
c2.metric(f"Low end ({level}% range)", f"{low:,.0f}")
c3.metric(f"High end ({level}% range)", f"{high:,.0f}")

for w in result["warnings"]:
    st.warning(w)

higher_than = (df[TARGET] < pred).mean() * 100
st.write(f"The predicted profit is higher than {higher_than:.0f}% of the 50 startups in the data.")

fig = px.histogram(df, x=TARGET, nbins=15, title="Profit of the 50 startups")
fig.add_vrect(x0=low, x1=high, fillcolor="red", opacity=0.15, line_width=0)
fig.add_vline(x=pred, line_dash="dash", line_color="red")
st.plotly_chart(fig)

with st.expander("How this works and what to be careful about"):
    st.write(
        "The model is a linear regression on the three spends, trained on 50 startups. "
        f"Its typical error in cross-validation is about {bundle['cv_rmse']:,.0f}. "
        "Almost all of the prediction comes from R&D spend. In my tests the marketing effect "
        "was very small, and the administration effect was not clearly different from zero."
    )
    coef = pd.DataFrame({"profit change per extra dollar spent": bundle["coefficients"]})
    st.dataframe(coef.round(3))
    st.write(
        "These numbers are links found in 50 startups, not proof of cause. "
        "The shaded range and the warnings are there so you do not read the result as exact."
    )
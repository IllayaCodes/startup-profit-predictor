"""Startup Profit Predictor (Streamlit app), version 2."""
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
out_of_range = bool(result["warnings"])

c1, c2, c3 = st.columns(3)
c1.metric("Predicted profit", f"{pred:,.0f}")
c2.metric(f"Low end ({level}% range)", f"{low:,.0f}")
c3.metric(f"High end ({level}% range)", f"{high:,.0f}")

if out_of_range:
    st.error(
        "Some inputs are outside the data this model learned from, "
        "so the numbers above are not reliable."
    )
    for w in result["warnings"]:
        st.warning(w)
else:
    higher_than = (df[TARGET] < pred).mean() * 100
    st.write(f"The predicted profit is higher than {higher_than:.0f}% of the 50 startups in the data.")

    fig = px.histogram(df, x=TARGET, nbins=15, title="Profit of the 50 startups")
    fig.add_vrect(x0=low, x1=high, fillcolor="red", opacity=0.15, line_width=0)
    fig.add_vline(x=pred, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

st.subheader("How your budget split compares")
total_in = sum(inputs.values())
if total_in > 0:
    user_share = {c: inputs[c] / total_in * 100 for c in SPEND_COLS}
    shares = df[SPEND_COLS].div(df[SPEND_COLS].sum(axis=1), axis=0) * 100
    top = df[TARGET] > df[TARGET].quantile(0.75)

    mix = pd.DataFrame({
        "Your budget": pd.Series(user_share),
        "Top 25% by profit (average)": shares[top].mean(),
        "Other startups (average)": shares[~top].mean(),
    })
    mix.index = [LABELS[c] for c in mix.index]
    mix.index.name = "Spend"
    long = mix.reset_index().melt(
        id_vars="Spend", var_name="Group", value_name="Share of total spend (%)"
    )
    fig2 = px.bar(long, x="Spend", y="Share of total spend (%)", color="Group", barmode="group")
    st.plotly_chart(fig2)

    top_total = df.loc[top, SPEND_COLS].sum(axis=1).mean()
    rest_total = df.loc[~top, SPEND_COLS].sum(axis=1).mean()
    st.caption(
        f"The top group is the {int(top.sum())} startups with the highest profit. They also spend more "
        f"in total (about {top_total:,.0f} on average, against {rest_total:,.0f} for the others), "
        "so the split and the amount go together. This shows what they did in this data, "
        "not what you should do."
    )
else:
    st.info("Enter at least one spend above zero to see the comparison.")

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
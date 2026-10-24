# Step 0: Ideation (Startup Profit Predictor)

---

## 1. Problem statement

Given **a startup's spend on R&D, Administration and Marketing**, estimate **its profit** so that **a founder or student** can **see how their budget mix compares with 50 other startups and explore what a different split is associated with**.

## 2. Who is it for, and what will they do with it?

- **User:** a founder, student or analyst who wants a quick, explainable sanity check of a budget split.
- **Decision:** whether their allocation looks unusual compared with the startups in the dataset, and which direction to explore.
- **What would make them trust the result:**
  - a prediction *range*, not a single number
  - a percentile against the 50 startups
  - a plain explanation of which spend drives the estimate
  - a warning when their inputs are outside the range the model has seen

This is a learning and demo project, not a tool for real funding decisions.

## 3. Questions the data should answer

1. Which spend category moves with profit the most?
2. Do R&D and Marketing overlap so much that their separate effects can't be told apart?
3. Is Administration spend associated with profit at all?
4. Do startups with a zero in some spend category earn less, and are there enough of them to say so?
5. Do the top 25% by profit spend a different *share* of their budget than the rest, or just a bigger *amount*?
6. Is the relationship between spend and profit roughly straight, or curved?
7. How much better is a complex model (Extra Trees, Gradient Boosting) than a simple linear one, once we use cross-validation?
8. How uncertain is a single prediction?

## 4. Hypotheses (and what would disprove each)

| # | Hypothesis | Evidence that would disprove it |
|---|------------|----------------------------------|
| H1 | R&D spend has the strongest positive association with profit. | Marketing's Spearman rho is equal to or higher than R&D's, or their 95% bootstrap intervals overlap heavily. |
| H2 | R&D and Marketing are correlated enough to blur their separate effects. | VIF for both stays below about 2 and their correlation is weak. |
| H3 | Administration has little or no association with profit. | Its 95% bootstrap interval for Spearman rho excludes 0 and is comparable to Marketing's. |
| H4 | Zero-spend startups earn less on average. | Their mean profit is similar to the rest, or the difference disappears when the group is this small (report n). |
| H5 | A linear model is nearly as good as tree ensembles on 50 rows. | A tree model beats it by more than the fold-to-fold standard deviation in repeated cross-validation. |
| H6 | Top performers differ in *amount* spent more than in *share* of budget. | Spend shares show a clear, significant relationship with profit on their own. |

## 5. Data inventory and the "available at prediction time" test

| Column | Meaning | Known when the user asks for a prediction? | Allowed as a model input? |
|--------|---------|---------------------------------------------|----------------------------|
| rd_spend | Spend on research and development | Yes, the user enters their planned or actual figure | Yes |
| admin_spend | Spend on administration | Yes | Yes |
| marketing_spend | Spend on marketing | Yes | Yes |
| profit | Profit of the startup | No, this is what we predict | **No, this is the target** |

**Allowed derived features** (computed from the three spends only): total spend, each category's share of total, R&D-to-Marketing ratio, log transforms.

**Analysis-only quantities** (need profit, never model inputs): profit per dollar of each category, "ROI", efficiency rank, profit rank.

**Other limits of the data:**
- only 50 rows, one table, no time column
- the period the spend and profit cover is not stated
- no industry, stage, location or year, so we can't say how typical these startups are
- the data is a well-known teaching dataset, so conclusions are about this sample

## 6. Success criteria

**Model:**
- **Baseline to beat:** linear regression on R&D spend alone, scored with repeated 5-fold cross-validation (for example, 10 repeats).
- **Target:** a more complex model is only chosen if it beats the baseline by more than the cross-validation standard deviation. If not, we keep the simpler, explainable model.
- **Metrics reported:** R², RMSE and MAE as mean ± standard deviation across folds.
- **Intervals:** prediction ranges derived from cross-validated errors, with coverage checked (a nominal 90% range should contain roughly 90% of held-out values).

**App:**
- **Must have:**
  - uses exactly the same feature function as training
  - shows a prediction range
  - warns on inputs outside the training range
  - shows the user's percentile against the 50 startups
  - allocation benchmarks that come from the data
- **Must not do:**
  - use hard-coded "industry averages" that aren't in the data
  - present what-if results as causal
  - crash on zero spend or empty inputs

## 7. Scope

**MVP (build first):**
- data audit and EDA scripts
- one shared `features.py` with leak-free features
- cross-validated model comparison and a saved model
- Streamlit app with prediction, range, allocation chart and percentile

**Stretch (only after the MVP works):**
- what-if sliders with the "association, not cause" label
- permutation importance or SHAP view
- PDF or CSV report export
- deployment on Streamlit Community Cloud or Hugging Face Spaces
- a few `pytest` tests (for example, the feature function never reads `profit`)

**Non-goals:**
- real investment advice
- predicting for startups far outside the data range
- deep learning or heavy hyperparameter searches on 50 rows

## 8. Risks and honest limitations

- **Small sample:** with 50 rows, results move a lot between splits, so we use repeated cross-validation and report spread.
- **Correlated inputs:** R&D and Marketing may move together, so individual coefficients can be unstable even when predictions are fine.
- **Association is not causation:** the data is observational, so "increase R&D by 20%" shows what similar startups look like, not what would happen to this one.
- **Extrapolation:** tree models can't extend beyond the training range, and linear models can extend it unrealistically. The app warns outside the range.
- **Tiny subgroups:** conclusions about the zero-spend rows rest on a handful of startups and must be labelled as anecdotal.
- **Leakage:** any feature built from profit invalidates the evaluation. Prevent it with the prediction-time test in Section 5.

## 9. Decisions made today

- Features will be spend-only. Profit-based ratios are analysis only.
- One feature function is used in training and in the app.
- We evaluate with repeated cross-validation, not a single 10-row split.
- The simplest model that performs about as well as the best is the one we ship.
- App advice and benchmarks are derived from the dataset, not hard-coded.
- The project is framed as a learning and demo tool.

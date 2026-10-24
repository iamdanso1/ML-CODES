# Forecasting Petroleum Product Prices in Ghana

Thesis codebase: a like-for-like comparison of **SARIMA**, **Random Forest**,
**XGBoost** and **LSTM** on National Petroleum Authority price data, benchmarked
against a random walk.

---

## Quick start

```bash
pip install -r requirements.txt
jupyter lab notebooks/Ghana_Petroleum_Price_Forecasting.ipynb
```

Run the notebook top to bottom. It writes every figure to `outputs/figures/`
(300 dpi PNG plus a vector PDF) and every table to `outputs/tables/`
(CSV plus a LaTeX version of the headline table).

To run the whole study on a different product, change one line in the
configuration cell:

```python
PRIMARY_TARGET = "PMS_price"      # or AGO_price, LPG_price, RFO_price, ATK_ghs, ...
```

---

## Layout

```
data/                     NPA workbooks, BoG file, LI 2186, CPI bulletin  (inputs)
data/processed/           tidy panel written by the ETL                   (generated)
notebooks/
  Ghana_Petroleum_Price_Forecasting.ipynb    the deliverable
  thesis_pipeline_part{1,2,3}.py             its source, in py:percent format
src/
  npa_etl.py              parses the NPA workbook into a tidy panel
  forecasting.py          features, metrics, the four models, the evaluation protocol
  thesis_viz.py           figure theme and chart helpers
  build_notebook.py       assembles (and optionally executes) the notebook
outputs/figures/          22 thesis-ready figures                         (generated)
outputs/tables/           metrics, diagnostics, forecasts                 (generated)
```

Editing the `.py` parts and rebuilding keeps the notebook reviewable in git:

```bash
python src/build_notebook.py --execute
```

---

## What the data actually supports

The thesis proposal lists four sources. Only one is present in `data/`:

| Source | Variable | Status |
|---|---|---|
| NPA | Product prices and their ex-refinery / tax / margin components, 1989-2026 | **available** |
| Bank of Ghana | Monthly USD/GHS rate | **available** — carried on every row of the NPA workbook |
| Bank of Ghana | `Interbank FX Rates Historical.xlsx` | unusable: 2 days of data, not a series |
| Ghana Statistical Service | Monthly CPI / inflation | **missing** — only one PDF bulletin (June 2012) |
| NPA | Monthly consumption volumes (PMS / AGO / LPG) | **missing** — no volume file |
| NPA | LI 2186 pricing formula | available as a reference document |

So the dependent variable is a **price**, not a consumed volume, and inflation
cannot enter as a regressor. Section 1.1 of the notebook states this in the
output itself. If the CPI series and the volume series are obtained later they
join the panel on the month index and nothing else needs changing.

### The panel

`src/npa_etl.py` turns the NPA workbook — a human-formatted spreadsheet with
merged product headers and a different layout per sheet — into a tidy panel:

| Series | Months | Span |
|---|---|---|
| Residual Fuel Oil | 224 | Sep 2007 – Aug 2026 |
| Premix | 212 | Jan 2009 – Aug 2026 |
| Aviation kerosene, marine gas oil, gas oil (mines/rig) | 188 | Jan 2011 – Aug 2026 |
| Petrol (PMS), Diesel (AGO), LPG, Kerosene — ex-pump | 90 | Sep 2007 – Jun 2015 |

The ex-pump series for the four consumer products stop in June 2015, when
deregulation ended NPA publication of them; only the regulated products
continue to 2026. The three internationally-quoted series are also expressed in
GHS per litre using the NPA's own reference rate. Each pricing-window quote is
converted using the FX rate recorded for that same window before the converted
quotes are averaged by month; multiplying monthly mean prices by monthly mean
FX rates can give a different result.

---

## Method

**Target.** Trees and the LSTM are given the **log return**, not the price
level: a tree cannot predict outside the range of its training targets, so
handing it a series that has tripled since training guarantees failure for
reasons unrelated to the model. Levels are rebuilt from the last observed
price. SARIMA reaches the same place through differencing.

**Information set.** Every feature is dated `t-1` or earlier, so all four models
condition on exactly what a genuine one-step-ahead forecast would know.

**Protocol.**

- chronological split, final 24 months held out and never trained on
- headline: one step ahead, fitted on the training span, hold-out filtered through without refitting
- secondary: rolling-origin recursive forecasts at horizons 1–12, refitted at each origin
- benchmarks: random walk and seasonal naive
- metrics: RMSE, MAE, MAPE, sMAPE, MASE, R², Theil's U2, bias, directional accuracy
- significance: Diebold-Mariano with the Harvey-Leybourne-Newbold correction
- replication: the whole benchmark re-run on every product series

**Why Theil's U2 and MASE carry the argument.** R² against a trending price
level is easy to inflate and says little. Both of these compare against the
random walk directly — below 1 means the model beat it.

---

## Latest corrected run

The notebook was successfully executed on 24 October 2026 after applying the
future-exogenous leakage fix, excluding zero-valued price quotes, and using the
NPA exchange rate for each pricing window before monthly averaging. The notebook
contains 82 cells, completed with no cell errors, and the outputs in `outputs/`
were regenerated by that run. See `outputs/tables/summary.txt` for the concise
results and `outputs/tables/headline_metrics.csv`,
`outputs/tables/multi_horizon_profile.csv`, and
`outputs/tables/diebold_mariano_pvalues.csv` for the supporting tables.

The corrected headline results show only modest one-step improvement: Random
Forest has the lowest RMSE (1.5144 GHS/L versus 1.5737 for the random walk), but
the 24-month Diebold–Mariano tests do not show statistically significant
differences at the 5% level. Do not claim statistically established model
superiority. These results concern an export-quoted Gas Oil (Mines/Rig) price
converted to GHS/L, not domestic retail diesel price and not petroleum demand.

## Reproducibility

`RANDOM_STATE = 42` seeds every random process. The LSTM additionally fits five
seeds and averages their predictions, so the reported result does not depend on
one initialisation; the notebook plots the spread between seeds so the reader
can see how much that mattered.

Hyper-parameters are tuned by randomised search over a `TimeSeriesSplit`, where
each validation fold is strictly later than the data used to fit it.

---

## Environment

The corrected notebook was successfully executed in this audit environment on
Python **3.13.5**, pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0, XGBoost
3.1.3, statsmodels 0.14.6 and PyTorch 2.10.0 (CPU). The requirements file uses
compatible version ranges rather than pinning these exact versions. For a fresh
reproduction, create a clean virtual environment and install `requirements.txt`
before opening or executing the notebook.

A complete run takes roughly **13–15 minutes** in this environment because it
includes SARIMA order search, time-series hyperparameter tuning, the LSTM seed
ensemble, 24 rolling origins and the 12-series replication. Runtime depends on
the CPU and installed library versions.


## Supplementary validation scripts

The package includes two supplementary scripts used for additional checks reported in the revised thesis:

- `python sensitivity_checks.py` — interpolation sensitivity, feature ablation, MASE scaling, and paired Random Forest-versus-naive checks.
- `python classical_benchmarks.py` — additional random-walk-with-drift, ETS, and ARIMAX benchmark comparisons.

Both scripts use paths relative to this project folder. Results are saved under `outputs/tables/`. Run the main notebook first if you want to regenerate the primary figures and tables. These supplementary checks are separate from the main notebook workflow.

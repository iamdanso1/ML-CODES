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
GHS per litre using the NPA's own reference rate.

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

## Results as currently configured

Primary series: **Gas Oil (mines/rig) ex-depot, GHS/litre**, Jan 2011 – Aug 2026,
24-month hold-out. One step ahead:

| Model | RMSE | MAPE | MASE | Theil U | R² |
|---|---|---|---|---|---|
| Random walk | 1.572 | 8.87% | 0.708 | 1.000 | 0.658 |
| Seasonal naive | 3.947 | 30.33% | 2.245 | 2.516 | −1.155 |
| SARIMA | 1.541 | 9.19% | 0.737 | **0.981** | 0.671 |
| **Random Forest** | **1.468** | **8.85%** | **0.707** | **0.933** | **0.702** |
| XGBoost | 1.523 | 8.86% | 0.708 | **0.968** | 0.679 |
| LSTM | 1.587 | 9.19% | 0.732 | 1.007 | 0.651 |
| Ensemble | 1.521 | 8.99% | 0.718 | **0.967** | 0.680 |

Four of the five models beat the random walk; Random Forest wins on every
accuracy measure. Across the 12 product series, **10 are beaten by at least one
model**, and each of the four methodologies wins somewhere — SARIMA takes LPG
(U 0.88), petrol and diesel; LSTM takes kerosene, marine gas oil and aviation
kerosene; Random Forest takes the gas oil series. For the three products the
proposal names, MAPE is 3.4–3.7%.

Two caveats the write-up should carry:

- **Seasonal strength is 0.04**, so AICc selects a non-seasonal order. SARIMA
  here is effectively ARIMA, and the notebook reports the AICc penalty for
  forcing seasonal terms rather than hiding the result.
- **Beyond one month the random walk is not beaten.** SARIMA tracks it to
  within 2% out to a year; the recursive tree forecasts drift and do worse.
  Section 13 shows the full horizon profile. That is a finding about how
  efficiently the NPA's pricing windows absorb information, not a modelling
  failure, and it is worth stating as such.

---

## Reproducibility

`RANDOM_STATE = 42` seeds every random process. The LSTM additionally fits five
seeds and averages their predictions, so the reported result does not depend on
one initialisation; the notebook plots the spread between seeds so the reader
can see how much that mattered.

Hyper-parameters are tuned by randomised search over a `TimeSeriesSplit`, where
each validation fold is strictly later than the data used to fit it.

---

## Environment

Developed and run on **Python 3.9.13** with PyTorch 2.5 (CUDA optional — the
LSTM falls back to CPU and takes well under a minute either way). See
`requirements.txt`.

Two things about this machine specifically:

- `python` on the `PATH` resolves to `C:\Python313`, which is a **broken
  install** — it has no `Lib/` directory, so it cannot even import `pip`. Don't
  use it. The working interpreter, and the one everything here was run with, is:

  ```bash
  C:/Users/Administrator/AppData/Local/Programs/Python/Python39/python.exe src/build_notebook.py --execute
  ```

- The `.venv/` in this repo was created against that Python 3.9 but has **no
  packages installed** (`include-system-site-packages = false`), so activating
  it will fail on the first import. Either install into it —

  ```bash
  .venv/Scripts/python.exe -m pip install -r requirements.txt
  ```

  — or keep using the system 3.9 interpreter, which already has every
  dependency.

A full run (ETL, EDA, SARIMA order search, two randomised searches, the LSTM
seed ensemble, 24 rolling origins and the 12-series replication) takes about
**11 minutes**.

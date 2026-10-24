# %% [markdown]
# # Forecasting Petroleum Product Prices in Ghana
#
# ### A comparative evaluation of SARIMA, Random Forest, XGBoost and LSTM
#
# ---
#
# **Data:** National Petroleum Authority (NPA) *Historical Trend of Petroleum
# Products Prices*, 1989 - August 2026, together with the cedi/dollar reference
# rate the NPA applies in each pricing window.
#
# **What this notebook does**
#
# 1. Parses the NPA workbook into a clean monthly panel of prices and price components.
# 2. Establishes the statistical character of the series (trend, seasonality, stationarity, volatility).
# 3. Fits the four models in the thesis methodology on an identical information set.
# 4. Evaluates them against a random-walk benchmark with scale-free metrics and a
#    Diebold-Mariano test, at one step ahead and across horizons 1-12.
# 5. Replicates the whole comparison across every product in the dataset.
# 6. Produces a 12-month ex-ante forecast and writes every figure and table to disk.
#
# **How to re-run it for a different product:** change `PRIMARY_TARGET` in the
# configuration cell. Nothing else needs editing.

# %% [markdown]
# ## 0. Environment and configuration

# %%
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import npa_etl
import forecasting as fc
import thesis_viz as viz
from npa_etl import PRODUCT_LABELS, POLICY_FIELDS, product_columns

viz.setup_theme()
FIG_DIR = ROOT / "outputs" / "figures"
TBL_DIR = ROOT / "outputs" / "tables"
for d in (FIG_DIR, TBL_DIR, ROOT / "data" / "processed"):
    d.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: "%.4f" % v)

print("python     ", sys.version.split()[0])
print("pandas     ", pd.__version__)
print("numpy      ", np.__version__)
import sklearn, xgboost, statsmodels, torch
print("scikit-learn", sklearn.__version__)
print("xgboost    ", xgboost.__version__)
print("statsmodels", statsmodels.__version__)
print("torch      ", torch.__version__, "| CUDA:", torch.cuda.is_available())

# %%
# --------------------------------------------------------------------------
# Study configuration -- the only cell that normally needs editing
# --------------------------------------------------------------------------

PRIMARY_TARGET = "AGO_MINES_ghs"   # export-quoted gas oil (mines/rig), converted to GHS per litre
TARGET_LABEL = "Gas Oil (Mines/Rig) export-quoted price converted to GHS/L"

SEASON = 12          # monthly data
TEST_MONTHS = 24     # hold-out length for the headline comparison
MAX_HORIZON = 12     # horizons profiled in the multi-step analysis
N_ORIGINS = 24       # rolling origins in the multi-horizon study
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# Market drivers: genuinely unknown at forecast time, so they enter lagged only.
MARKET_DRIVERS = [
    "fx_ghs_usd",            # cedi per US dollar, NPA reference rate
    "ATK_price",             # aviation turbine kerosene, USD/litre
    "MGO_FOREIGN_price",     # marine gasoil (foreign), USD/litre
    "AGO_MINES_EXP_price",   # gas oil mines/rig, USD/litre
    "RFO_price",             # residual fuel oil, GHS/litre
]

print("primary target :", PRIMARY_TARGET)
print("hold-out       :", TEST_MONTHS, "months")
print("horizons       : 1 -", MAX_HORIZON)

# %% [markdown]
# ## 1. Data ingestion
#
# The NPA workbook is a human-formatted spreadsheet rather than a data file: a
# banner, a merged product header, a field header, a units row, then the
# observations, repeated with a different layout on each sheet. `src/npa_etl.py`
# handles that - it forward-fills the merged product names across the header,
# maps each `(product, field)` pair to a column, converts pesewas and US cents to
# cedis and dollars, and stacks the sheets into one panel.

# %%
panel = npa_etl.build_panel(ROOT / "data")
npa_etl.save_panel(panel, ROOT / "data" / "processed")

windows = panel["wide"]      # NPA pricing-window frequency (1st and 16th)
monthly = panel["monthly"]   # month-end resample
coverage = panel["coverage"]

print("pricing windows :", windows.shape, "|", windows.index.min().date(), "->", windows.index.max().date())
print("monthly panel   :", monthly.shape)
coverage[["label", "series", "currency", "unit", "n_windows", "n_months", "first", "last"]]

# %% [markdown]
# ### 1.1 Data availability audit
#
# The thesis proposal lists four data sources. Only one of them is present in
# `data/`, and the notebook is explicit about that rather than silently
# substituting: the modelling below uses what the repository actually contains.

# %%
audit = pd.DataFrame(
    [
        ("NPA", "Petroleum product prices & price components",
         "AVAILABLE", "Historical-Trend-of-Petroleum-Products-Prices ... .xlsx, 1989-2026"),
        ("Bank of Ghana", "Monthly USD/GHS reference rate",
         "AVAILABLE (embedded)", "Carried on every row of the NPA workbook; the rate the NPA itself applies"),
        ("Bank of Ghana", "Interbank FX time series",
         "NOT USABLE", "'Interbank FX Rates Historical.xlsx' holds 2 days (30-31 Dec 2025), not a series"),
        ("Ghana Statistical Service", "Monthly CPI / headline inflation",
         "MISSING", "Only one PDF bulletin (June 2012) is present - no monthly series to merge"),
        ("NPA", "Monthly consumption volumes (PMS / AGO / LPG)",
         "MISSING", "No volume file in data/ - consumption cannot be modelled from this repository"),
        ("NPA / CBOD", "Pricing formula regulations",
         "AVAILABLE (reference)", "LI 2186 PDF - documents the ex-refinery + taxes + margins build-up"),
    ],
    columns=["Source", "Variable", "Status", "Note"],
)
audit.style.hide(axis="index")

# %% [markdown]
# **Consequence for the study design.** Consumption volumes and the CPI series
# are absent, so the dependent variable is the **price** of a petroleum product,
# not its consumed volume, and inflation cannot enter as a regressor. The
# exchange rate survives, because the NPA records the rate it applied in each
# pricing window. If the GSS CPI series and the NPA volume series are obtained
# later, they join the panel on the month index and the rest of the notebook
# runs unchanged.

# %% [markdown]
# ## 2. Panel construction
#
# Three of the long series are quoted in US cents per litre because they are
# export-parity products. Multiplying by the NPA's own reference rate expresses
# them in cedis per litre. The converted Gas Oil (Mines/Rig) series remains an
# export-quoted price indicator; it is not the domestic consumer diesel price.

# %%
panel_m = monthly.copy()
# Convert each export-quoted price using the NPA reference FX rate recorded
# for that same pricing window, then average the converted values by month.
# Multiplying monthly mean price by monthly mean FX would not, in general,
# equal the mean of the window-level converted prices.
panel_w = windows.copy()
for p in ["ATK", "MGO_FOREIGN", "AGO_MINES_EXP"]:
    col = p + "_price"
    if col in panel_w.columns:
        converted_col = p.replace("_EXP", "") + "_ghs"
        panel_w[converted_col] = panel_w[col] * panel_w["fx_ghs_usd"]
        panel_m[converted_col] = panel_w[converted_col].resample("ME").mean().reindex(panel_m.index)

# Save a reproducible audit of zero-valued price quotes excluded by the ETL.
quality_audit = panel["quality_audit"].copy()
if not quality_audit.empty:
    quality_audit["date"] = pd.to_datetime(quality_audit["date"]).dt.strftime("%Y-%m-%d")
    special = quality_audit["date"].eq("2014-08-01") & quality_audit["product"].eq("AGO_MINES_EXP")
    if special.any():
        aug_value = panel_m.loc[pd.Timestamp("2014-08-31"), "AGO_MINES_ghs"]
        quality_audit.loc[special, "effect"] = (
            "Excluded before monthly averaging; August 2014 converted monthly value is "
            + f"GHS {aug_value:.4f}/L based on the positive pricing-window quote."
        )
    quality_audit.to_csv(TBL_DIR / "data_quality_audit.csv", index=False)

# A cedi-converted series carries the name of the product it was derived from,
# which is what the feature builder needs to find its price components.
GHS_BASE = {"AGO_MINES_ghs": "AGO_MINES_EXP", "MGO_FOREIGN_ghs": "MGO_FOREIGN",
            "ATK_ghs": "ATK"}

def base_product(col):
    return GHS_BASE.get(col, col.rsplit("_", 1)[0])

SERIES_LABELS = {
    "AGO_MINES_ghs": "Gas Oil (mines/rig), export-quoted, converted to GHS/L",
    "MGO_FOREIGN_ghs": "Marine Gas Oil, GHS/L",
    "ATK_ghs": "Aviation Kerosene, GHS/L",
    "AGO_MINES_EXP_price": "Gas Oil (mines/rig), USD/L",
    "MGO_FOREIGN_price": "Marine Gas Oil, USD/L",
    "ATK_price": "Aviation Kerosene, USD/L",
    "RFO_price": "Residual Fuel Oil, GHS/L",
    "PREMIX_price": "Premix, GHS/L",
    "PMS_price": "Petrol PMS ex-pump, GHS/L",
    "AGO_price": "Diesel AGO ex-pump, GHS/L",
    "LPG_price": "LPG ex-pump, GHS/kg",
    "KERO_price": "Kerosene ex-pump, GHS/L",
}

STUDY_SERIES = [s for s in SERIES_LABELS if s in panel_m.columns]
summary = pd.DataFrame({
    "label": [SERIES_LABELS[s] for s in STUDY_SERIES],
    "n_months": [int(panel_m[s].notna().sum()) for s in STUDY_SERIES],
    "first": [panel_m[s].first_valid_index() for s in STUDY_SERIES],
    "last": [panel_m[s].last_valid_index() for s in STUDY_SERIES],
    "mean": [panel_m[s].mean() for s in STUDY_SERIES],
    "sd": [panel_m[s].std() for s in STUDY_SERIES],
    "monthly_vol_%": [np.log(panel_m[s].dropna()).diff().std() * 100 for s in STUDY_SERIES],
}, index=STUDY_SERIES)
summary.sort_values("n_months", ascending=False)

# %% [markdown]
# ### 2.1 The primary series

# %%
own_cols = [c for c in product_columns(panel_m.columns, base_product(PRIMARY_TARGET))
            if c != PRIMARY_TARGET]
exog_cols = list(dict.fromkeys(own_cols + [c for c in MARKET_DRIVERS if c != PRIMARY_TARGET]))

data = fc.prepare_series(panel_m, PRIMARY_TARGET, exog_cols)
y = data[PRIMARY_TARGET]

print("series      :", SERIES_LABELS.get(PRIMARY_TARGET, PRIMARY_TARGET))
print("span        :", y.index.min().date(), "->", y.index.max().date(), "(%d months)" % len(y))
print("imputed     :", data.attrs["imputed_months"], "months with no NPA pricing window")
print("exogenous   :", [c for c in data.columns if c != PRIMARY_TARGET])
print()
y.describe().to_frame("value").T

# %% [markdown]
# ## 3. Exploratory data analysis
#
# ### 3.1 The long-run picture

# %%
fig, ax = plt.subplots(figsize=(11, 5))
show = ["AGO_MINES_ghs", "MGO_FOREIGN_ghs", "ATK_ghs", "RFO_price", "PREMIX_price"]
show = [s for s in show if s in panel_m.columns]
end_labels = []
for i, s in enumerate(show):
    ser = panel_m[s].dropna()
    ax.plot(ser.index, ser.values, color=viz.CATEGORICAL[i], label=SERIES_LABELS[s],
            linewidth=2.0 if s == PRIMARY_TARGET else 1.4,
            alpha=1.0 if s == PRIMARY_TARGET else 0.75)
    end_labels.append((ser.index[-1], ser.iloc[-1], SERIES_LABELS[s].split(",")[0],
                       viz.CATEGORICAL[i]))
ax.set_yscale("log")
ax.set_ylabel("GHS per litre (log scale)")
ax.set_xlim(panel_m.index.min(), panel_m.index.max() + pd.DateOffset(months=62))
viz.titled(ax, "Ghanaian petroleum prices, 2007-2026",
           "Log scale: a constant slope is a constant growth rate. Every series is dominated by cedi depreciation.")
ax.legend(loc="upper left", ncol=2)
# Three of these five end within a few pesewas of each other, so the endpoint
# labels have to be separated explicitly or they overprint.
viz.label_ends(ax, end_labels, min_gap=16)
viz.annotate_source(fig, "Source: NPA Historical Trend of Petroleum Products Prices; own calculations.")
viz.save_fig(fig, "01_long_run_prices", FIG_DIR)
plt.show()

# %% [markdown]
# ### 3.2 The primary series and its hold-out window

# %%
test_start_m = y.index[-TEST_MONTHS]

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
ax = axes[0]
ax.plot(y.index, y.values, color=viz.CATEGORICAL[0], linewidth=2.0)
viz.shade_span(ax, test_start_m, y.index[-1], "hold-out (%d months)" % TEST_MONTHS)
viz.label_end(ax, y.index[-1], y.iloc[-1], "%.2f" % y.iloc[-1], viz.CATEGORICAL[0])
ax.set_ylabel(TARGET_LABEL)
viz.titled(ax, SERIES_LABELS.get(PRIMARY_TARGET, PRIMARY_TARGET),
           "Level. The shaded window is never seen by any model during training.")

ax2 = axes[1]
r = np.log(y).diff() * 100
ax2.axhline(0, color=viz.INK["axis"], linewidth=0.8)
ax2.bar(r.index, r.values, width=22,
        color=np.where(r.values >= 0, viz.CATEGORICAL[0], viz.CATEGORICAL[1]))
viz.shade_span(ax2, test_start_m, y.index[-1])
ax2.set_ylabel("monthly log change, %")
viz.titled(ax2, "Month-on-month change",
           "Blue = increase, orange = decrease. Volatility clusters around the 2014-16 and 2022 cedi episodes.")
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
viz.save_fig(fig, "02_primary_series", FIG_DIR)
plt.show()

# %% [markdown]
# ### 3.3 Exchange-rate pass-through
#
# Two quantities on very different scales must never share a plot with two y-axes:
# the alignment of the scales would invent a relationship. Both are indexed to
# 100 at the start of the sample and drawn on one axis instead.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
base = data.loc[data.index[0]]
ends = []
for i, (col, name) in enumerate([(PRIMARY_TARGET, "Fuel price (GHS)"),
                                 ("fx_ghs_usd", "GHS per USD")]):
    idx = data[col] / base[col] * 100
    ax.plot(idx.index, idx.values, color=viz.CATEGORICAL[i], label=name)
    ends.append((idx.index[-1], idx.iloc[-1], name, viz.CATEGORICAL[i]))
ax.set_xlim(data.index[0], data.index[-1] + pd.DateOffset(months=52))
viz.label_ends(ax, ends)
ax.set_ylabel("index, first month = 100")
ax.legend(loc="upper left")
viz.titled(ax, "Both series indexed to 100",
           "One axis, one scale: the gap between the lines is the real-terms move.")

ax = axes[1]
dl_price = np.log(data[PRIMARY_TARGET]).diff()
dl_fx = np.log(data["fx_ghs_usd"]).diff()
lags = range(0, 7)
corrs = [dl_price.corr(dl_fx.shift(L)) for L in lags]
bars = ax.bar(list(lags), corrs, color=viz.CATEGORICAL[0], width=0.62)
best = int(np.nanargmax(np.abs(corrs)))
bars[best].set_color(viz.CATEGORICAL[1])
ax.annotate("strongest at lag %d (r = %.2f)" % (best, corrs[best]),
            xy=(best, corrs[best]), xytext=(0, 8), textcoords="offset points",
            ha="center", color=viz.CATEGORICAL[1], fontsize=9)
ax.axhline(0, color=viz.INK["axis"], linewidth=0.8)
ax.set_xlabel("lag of the exchange-rate change, months")
ax.set_ylabel("correlation with the price change")
viz.titled(ax, "Pass-through timing",
           "Cross-correlation of monthly log changes.")
fig.tight_layout()
viz.save_fig(fig, "03_fx_passthrough", FIG_DIR)
plt.show()

# %% [markdown]
# ### 3.4 The pricing formula
#
# LI 2186 builds the published price as *ex-refinery price + taxes and levies +
# regulated margins*. Splitting the series that way shows which component
# actually moves, and therefore what a forecasting model has to track.

# %%
comp_product = "PMS"        # the domestic build-up is published in full for PMS
comp_cols = product_columns(monthly.columns, comp_product, ("ex_ref", "taxes", "margins"))
comp = monthly[comp_cols].dropna()

if len(comp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    labels = ["Ex-refinery price", "Taxes & levies", "Regulated margins"]
    ax.stackplot(comp.index, [comp[c].values for c in comp_cols],
                 labels=labels, colors=[viz.CATEGORICAL[0], viz.CATEGORICAL[1], viz.CATEGORICAL[2]],
                 edgecolor=viz.INK["surface"], linewidth=1.4)
    ax.set_ylabel("GHS per litre")
    ax.legend(loc="upper left")
    viz.titled(ax, "What a litre of petrol is made of",
               "NPA build-up for PMS, %s-%s." % (comp.index[0].year, comp.index[-1].year))

    ax = axes[1]
    share = comp.div(comp.sum(axis=1), axis=0) * 100
    for i, (c, lab) in enumerate(zip(comp_cols, labels)):
        ax.plot(share.index, share[c].values, color=viz.CATEGORICAL[i], label=lab)
        viz.label_end(ax, share.index[-1], share[c].iloc[-1], "%.0f%%" % share[c].iloc[-1],
                      viz.CATEGORICAL[i])
    ax.set_ylabel("share of the ex-pump price, %")
    ax.set_xlim(share.index[0], share.index[-1] + pd.DateOffset(months=14))
    ax.legend(loc="center left")
    viz.titled(ax, "Composition over time",
               "The ex-refinery share is the part that moves with world prices and the cedi.")
    fig.tight_layout()
    viz.save_fig(fig, "04_pricing_formula", FIG_DIR)
    plt.show()

# %% [markdown]
# ## 4. Time-series diagnostics
#
# ### 4.1 STL decomposition

# %%
from statsmodels.tsa.seasonal import STL

stl = STL(np.log(y), period=SEASON, robust=True).fit()
parts = [("Observed (log)", np.log(y), viz.INK["primary"]),
         ("Trend", stl.trend, viz.CATEGORICAL[0]),
         ("Seasonal", stl.seasonal, viz.CATEGORICAL[2]),
         ("Remainder", stl.resid, viz.CATEGORICAL[1])]

fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
for ax, (name, series, color) in zip(axes, parts):
    if name == "Remainder":
        ax.axhline(0, color=viz.INK["axis"], linewidth=0.8)
        ax.vlines(series.index, 0, series.values, color=color, linewidth=1.4)
    else:
        ax.plot(series.index, series.values, color=color, linewidth=1.8)
    ax.set_ylabel(name, fontsize=10)
seas_strength = max(0.0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
trend_strength = max(0.0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())
viz.titled(axes[0], "STL decomposition of the log price",
           "Trend strength %.2f   |   Seasonal strength %.2f   (0 = none, 1 = total)"
           % (trend_strength, seas_strength))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
viz.save_fig(fig, "05_stl_decomposition", FIG_DIR)
plt.show()

print("Trend strength    %.3f" % trend_strength)
print("Seasonal strength %.3f" % seas_strength)

# %% [markdown]
# ### 4.2 Stationarity
#
# ADF tests the null "there is a unit root"; KPSS tests the opposite null
# "the series is stationary". Reading them together is what settles the
# differencing order `d` for SARIMA, and confirms that the machine-learning
# models should be given returns rather than levels.

# %%
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_table(series_map):
    rows = []
    for name, s in series_map.items():
        s = s.dropna()
        adf = adfuller(s, autolag="AIC")
        kp = kpss(s, regression="c", nlags="auto")
        rows.append({
            "series": name, "n": len(s),
            "ADF stat": adf[0], "ADF p": adf[1],
            "ADF verdict": "stationary" if adf[1] < 0.05 else "unit root",
            "KPSS stat": kp[0], "KPSS p": kp[1],
            "KPSS verdict": "stationary" if kp[1] > 0.05 else "non-stationary",
        })
    return pd.DataFrame(rows).set_index("series")

stat_tbl = stationarity_table({
    "Price level": y,
    "Log price": np.log(y),
    "Log return (1st difference)": np.log(y).diff(),
    "Seasonal difference of log": np.log(y).diff(SEASON),
})
stat_tbl.to_csv(TBL_DIR / "stationarity_tests.csv")
stat_tbl

# %% [markdown]
# ### 4.3 Autocorrelation structure

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
for j, (name, series) in enumerate([("Log price", np.log(y)),
                                    ("Log return", np.log(y).diff().dropna())]):
    plot_acf(series, lags=36, ax=axes[0, j], color=viz.CATEGORICAL[0],
             vlines_kwargs={"colors": viz.CATEGORICAL[0]}, zero=False)
    plot_pacf(series, lags=36, ax=axes[1, j], method="ywm", color=viz.CATEGORICAL[0],
              vlines_kwargs={"colors": viz.CATEGORICAL[0]}, zero=False)
    axes[0, j].set_title("ACF - %s" % name, loc="left")
    axes[1, j].set_title("PACF - %s" % name, loc="left")
for ax in axes.ravel():
    for coll in ax.collections:
        coll.set_facecolor(viz.CATEGORICAL[0])
        coll.set_alpha(0.15)
    ax.set_xlabel("lag (months)")
fig.suptitle("")
fig.tight_layout()
viz.save_fig(fig, "06_acf_pacf", FIG_DIR)
plt.show()

# %% [markdown]
# ### 4.4 Volatility and the distribution of returns

# %%
from scipy import stats as sps

fig = plt.figure(figsize=(12, 4.8))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.7, 1, 1], wspace=0.32)

ax = fig.add_subplot(gs[0, 0])
rv = r.rolling(12).std()
ax.plot(rv.index, rv.values, color=viz.CATEGORICAL[0])
ax.fill_between(rv.index, 0, rv.values, color=viz.CATEGORICAL[0], alpha=0.10, linewidth=0)
hi = rv.idxmax()
ax.annotate("peak %.1f%%\n%s" % (rv.max(), hi.strftime("%b %Y")), xy=(hi, rv.max()),
            xytext=(-6, -28), textcoords="offset points", fontsize=9,
            color=viz.CATEGORICAL[1], ha="right")
ax.set_ylabel("12-month rolling sd of log change, %")
viz.titled(ax, "Volatility clustering", "Calm regimes and crisis regimes alternate.")

ax = fig.add_subplot(gs[0, 1])
rr = r.dropna()
ax.hist(rr, bins=28, color=viz.CATEGORICAL[0], alpha=0.85, edgecolor=viz.INK["surface"])
xs = np.linspace(rr.min(), rr.max(), 200)
ax.plot(xs, sps.norm.pdf(xs, rr.mean(), rr.std()) * len(rr) * (rr.max() - rr.min()) / 28,
        color=viz.CATEGORICAL[1], linewidth=1.8, label="normal fit")
ax.set_xlabel("monthly log change, %")
ax.legend()
viz.titled(ax, "Fat tails", "Skew %.2f, excess kurtosis %.2f" % (sps.skew(rr), sps.kurtosis(rr)))

ax = fig.add_subplot(gs[0, 2])
sps.probplot(rr, dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor(viz.CATEGORICAL[0])
ax.get_lines()[0].set_markeredgecolor(viz.CATEGORICAL[0])
ax.get_lines()[0].set_markersize(4)
ax.get_lines()[1].set_color(viz.CATEGORICAL[1])
ax.set_title("Normal Q-Q", loc="left")
ax.set_xlabel("theoretical quantiles"); ax.set_ylabel("observed")
fig.tight_layout()
viz.save_fig(fig, "07_volatility_distribution", FIG_DIR)
plt.show()

jb = sps.jarque_bera(rr)
print("Jarque-Bera: stat %.2f, p %.4g -> %s"
      % (jb[0], jb[1], "returns are not normal" if jb[1] < 0.05 else "normality not rejected"))

# %% [markdown]
# ### 4.5 Seasonality by calendar month

# %%
by_month = pd.DataFrame({"month": r.index.month, "chg": r.values}).dropna()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

ax = axes[0]
groups = [by_month.loc[by_month["month"] == m, "chg"].values for m in range(1, 13)]
bp = ax.boxplot(groups, patch_artist=True, widths=0.6, showfliers=False,
                medianprops={"color": viz.INK["primary"], "linewidth": 1.6},
                whiskerprops={"color": viz.INK["axis"]}, capprops={"color": viz.INK["axis"]})
for patch in bp["boxes"]:
    patch.set_facecolor(viz.CATEGORICAL[0])
    patch.set_alpha(0.35)
    patch.set_edgecolor(viz.CATEGORICAL[0])
ax.axhline(0, color=viz.INK["axis"], linewidth=0.8)
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("monthly log change, %")
viz.titled(ax, "Change by calendar month", "Boxes overlap the zero line in every month.")

ax = axes[1]
mean_by_month = by_month.groupby("month")["chg"].mean()
se = by_month.groupby("month")["chg"].sem()
ax.bar(mean_by_month.index, mean_by_month.values, color=viz.CATEGORICAL[0], width=0.62)
ax.errorbar(mean_by_month.index, mean_by_month.values, yerr=1.96 * se, fmt="none",
            ecolor=viz.INK["secondary"], elinewidth=1.0, capsize=3)
ax.axhline(0, color=viz.INK["axis"], linewidth=0.8)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylabel("mean change, %")
viz.titled(ax, "Mean change with 95% intervals",
           "No month is reliably different from zero - weak deterministic seasonality.")
fig.tight_layout()
viz.save_fig(fig, "08_seasonality", FIG_DIR)
plt.show()

# %% [markdown]
# ### 4.6 What moves with what

# %%
cm = np.log(data.where(data > 0)).diff().corr()
nice = [SERIES_LABELS.get(c, c.replace("_", " ")) for c in cm.columns]

# A diverging ramp is for data with two signs. These correlations are almost all
# positive, so a diverging map would render the whole matrix in one pole and
# waste half its range; fall back to the single-hue magnitude ramp in that case.
off_diag = cm.values[~np.eye(len(cm), dtype=bool)]
has_negative = np.nanmin(off_diag) < -0.1
cmap, vmin, vmax = ((viz.DIVERGING, -1, 1) if has_negative
                    else (viz.SEQ_BLUE, 0, 1))

fig, ax = plt.subplots(figsize=(7.6, 6.4))
im = ax.imshow(cm.values, cmap=cmap, vmin=vmin, vmax=vmax)
ax.set_xticks(range(len(nice))); ax.set_xticklabels(nice, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(nice))); ax.set_yticklabels(nice, fontsize=9)
for i in range(len(nice)):
    for j in range(len(nice)):
        v = cm.values[i, j]
        dark = abs(v) > 0.6 if has_negative else v > 0.62
        ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8,
                color=viz.INK["surface"] if dark else viz.INK["secondary"])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.78)
cb.set_label("correlation of monthly log changes", fontsize=9)
viz.titled(ax, "Co-movement of the drivers",
           "Diverging scale, grey at zero." if has_negative
           else "One hue, light to dark: every relationship here is positive.")
fig.tight_layout()
viz.save_fig(fig, "09_correlation_matrix", FIG_DIR)
plt.show()

# %% [markdown]
# ### 4.7 Near-duplicate drivers
#
# The NPA publishes several export products off a shared ex-refinery
# quotation, so two columns with different names can be the same series plus a
# constant duty. That is not leakage - both are lagged - but it inflates the
# apparent number of independent drivers and splits importance between twins,
# so the duplicates are found and reported rather than left to surprise a reader.

# %%
dup_rows = []
cols_c = list(cm.columns)
for i in range(len(cols_c)):
    for j in range(i + 1, len(cols_c)):
        rho = cm.iloc[i, j]
        if pd.notna(rho) and abs(rho) > 0.99:
            a, b = data[cols_c[i]], data[cols_c[j]]
            dup_rows.append({
                "series A": cols_c[i], "series B": cols_c[j],
                "corr of changes": rho,
                "max abs difference": float((a - b).abs().max()),
                "mean abs difference": float((a - b).abs().mean()),
            })
duplicates = pd.DataFrame(dup_rows)
if len(duplicates):
    print("Near-duplicate driver pairs (|r| > 0.99 on monthly log changes):\n")
    print(duplicates.round(4).to_string(index=False))
    print("\nA constant gap with a correlation of 1.00 means the two columns are the")
    print("same quotation separated by a fixed excise duty. Both are retained below -")
    print("the tree models split importance between them, which is flagged in 8.1.")
else:
    print("No driver pair exceeds |r| = 0.99; the feature set carries no exact twins.")

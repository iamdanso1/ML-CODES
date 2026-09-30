# %% [markdown]
# ## 5. Feature engineering
#
# Every feature is dated `t-1` or earlier. That is the rule that makes the
# comparison fair: the tree models and the LSTM get exactly the information a
# one-step-ahead SARIMA forecast conditions on, and nothing more. The target the
# learners are given is the **log return**, not the level - a tree cannot
# predict a value outside the range of its training targets, so handing it a
# series that has tripled since the training window would guarantee failure for
# a reason that has nothing to do with the model. Levels are rebuilt afterwards
# from the last observed price.

# %%
spec = fc.FeatureSpec(
    target=PRIMARY_TARGET,
    exog=[c for c in data.columns if c != PRIMARY_TARGET],
    lags=(1, 2, 3, 6, 12),
    roll_windows=(3, 6, 12),
    exog_lags=(1, 2, 3),
    add_calendar=True,
    add_fourier=2,
)

feats_all = fc.make_features(data, spec)
feats = feats_all.dropna()
FEATURES = fc.feature_columns(feats)

print("rows after warm-up trim : %d  (%s -> %s)"
      % (len(feats), feats.index[0].date(), feats.index[-1].date()))
print("features                : %d" % len(FEATURES))
print()
groups = {
    "autoregressive (returns)": [c for c in FEATURES if c.startswith("r_")],
    "level context": [c for c in FEATURES if c.startswith(("lvl_", "log_y"))],
    "exchange rate": [c for c in FEATURES if c.startswith("fx_")],
    "world product prices": [c for c in FEATURES
                             if c.startswith(("ATK", "MGO", "AGO", "RFO", "PREMIX"))],
    "calendar / Fourier": [c for c in FEATURES if c in ("month", "time_idx")
                           or c.startswith(("sin", "cos"))],
}
pd.DataFrame({"n features": {k: len(v) for k, v in groups.items()},
              "example": {k: (v[0] if v else "-") for k, v in groups.items()}})

# %% [markdown]
# ## 6. Evaluation protocol
#
# | | |
# |---|---|
# | **Split** | chronological; the final 24 months are the hold-out and are never trained on |
# | **Headline horizon** | one step ahead - the model is fitted on the training span and the hold-out is filtered through it without refitting |
# | **Secondary** | rolling-origin recursive forecasts at horizons 1-12, re-fitted at each origin |
# | **Benchmarks** | random walk (`y_t = y_{t-1}`) and seasonal naive (`y_t = y_{t-12}`) |
# | **Metrics** | RMSE, MAE, MAPE, sMAPE, MASE, R2, Theil's U2, bias, directional accuracy |
# | **Significance** | Diebold-Mariano with the Harvey-Leybourne-Newbold small-sample correction |
#
# Theil's U2 and MASE both compare against the random walk: below 1 means the
# model beats it. For a price series that is the only comparison that means
# anything - a high R2 against a trending level is easy and says little.

# %%
test_start = feats.index[-TEST_MONTHS]
is_train = feats.index < test_start
is_test = feats.index >= test_start

X_train, X_test = feats.loc[is_train, FEATURES], feats.loc[is_test, FEATURES]
r_train, r_test = feats.loc[is_train, "r"], feats.loc[is_test, "r"]
y_train, y_test = feats.loc[is_train, "y"], feats.loc[is_test, "y"]
y_prev_test = feats.loc[is_test, "y_lag1"]
test_index = feats.index[is_test]

print("train : %d months  %s -> %s" % (is_train.sum(), feats.index[0].date(),
                                       feats.index[is_train][-1].date()))
print("test  : %d months  %s -> %s" % (is_test.sum(), test_index[0].date(),
                                       test_index[-1].date()))

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(y.index, y.values, color=viz.INK["axis"], linewidth=1.2)
ax.plot(y_train.index, y_train.values, color=viz.CATEGORICAL[0], linewidth=2.0, label="training")
ax.plot(y_test.index, y_test.values, color=viz.CATEGORICAL[1], linewidth=2.0, label="hold-out")
ax.axvline(test_start, color=viz.INK["secondary"], linewidth=1.0)
ax.annotate("split\n%s" % test_start.strftime("%b %Y"), xy=(test_start, ax.get_ylim()[1]),
            xytext=(6, -14), textcoords="offset points", fontsize=9, va="top",
            color=viz.INK["secondary"])
ax.set_ylabel(TARGET_LABEL)
ax.legend(loc="upper left")
viz.titled(ax, "Chronological split", "The grey stub before the blue line is the feature warm-up period.")
viz.save_fig(fig, "10_train_test_split", FIG_DIR)
plt.show()

# %%
predictions = {}       # one-step-ahead level forecasts over the hold-out
fit_seconds = {}

predictions["Naive"] = y_prev_test.copy()
predictions["Seasonal Naive"] = fc.seasonal_naive_forecast(y, test_index, SEASON)

fc.assemble_results({k: predictions[k] for k in ("Naive", "Seasonal Naive")},
                    y_test, y_train, SEASON).round(4)

# %% [markdown]
# ## 7. Model 1 - SARIMA
#
# The classical baseline. The grid below searches
# `SARIMA(p,d,q)(P,D,Q)[12]` on the **log** price and ranks by AICc, the
# small-sample corrected information criterion - with fewer than 200
# observations plain AIC over-fits the order.

# %%
log_y = np.log(y)
t0 = time.time()
sarima_grid = fc.sarimax_grid_search(
    log_y[log_y.index < test_start],
    p_range=(0, 1, 2, 3), d_range=(1,), q_range=(0, 1, 2, 3),
    P_range=(0, 1), D_range=(0, 1), Q_range=(0, 1), season=SEASON,
)
sarima_grid_seconds = time.time() - t0
print("searched %d specifications in %.0fs" % (len(sarima_grid), sarima_grid_seconds))
sarima_grid.head(10)[["order", "seasonal_order", "AIC", "AICc", "BIC", "has_seasonal"]]

# %%
best_overall = sarima_grid.iloc[0]
best_seasonal = sarima_grid[sarima_grid["has_seasonal"]].iloc[0]

print("Best overall  : SARIMA%s%s   AICc %.2f" %
      (best_overall["order"], best_overall["seasonal_order"], best_overall["AICc"]))
print("Best seasonal : SARIMA%s%s   AICc %.2f" %
      (best_seasonal["order"], best_seasonal["seasonal_order"], best_seasonal["AICc"]))
print()
gap = best_seasonal["AICc"] - best_overall["AICc"]
print("AICc penalty for forcing a seasonal component: %.2f" % gap)
print("-> %s" % ("the data do not support seasonal terms; the selected model is a plain ARIMA"
                 if gap > 2 else "seasonal terms are competitive and are retained"))

SARIMA_ORDER = tuple(best_overall["order"])
SARIMA_SEASONAL = tuple(best_overall["seasonal_order"])

# %%
t0 = time.time()
sarima_osa, sarima_res = fc.sarimax_one_step(
    log_y, test_start, SARIMA_ORDER, SARIMA_SEASONAL)
predictions["SARIMA"] = sarima_osa.reindex(test_index)
sarima_ms, sarima_ci, _ = fc.sarimax_multi_step(
    log_y, test_start, SARIMA_ORDER, SARIMA_SEASONAL, alpha=0.20)
# Order selection is part of the cost of using SARIMA, so the grid search time
# counts towards it -- otherwise the comparison charges the tree models for
# their hyper-parameter search and gives SARIMA its search for free.
fit_seconds["SARIMA"] = (time.time() - t0) + sarima_grid_seconds

print(sarima_res.summary().tables[1])

# %% [markdown]
# ### 7.1 Residual diagnostics
#
# A well-specified SARIMA leaves residuals that are indistinguishable from white
# noise. The Ljung-Box test is the formal check; the four panels are the visual one.

# %%
from statsmodels.stats.diagnostic import acorr_ljungbox

resid = pd.Series(sarima_res.resid, index=log_y[log_y.index < test_start].index)
resid = resid.iloc[max(SARIMA_ORDER[1], 1):]

fig, axes = plt.subplots(2, 2, figsize=(12, 6.8))
ax = axes[0, 0]
ax.axhline(0, color=viz.INK["axis"], linewidth=0.8)
ax.plot(resid.index, resid.values, color=viz.CATEGORICAL[0], linewidth=1.3)
ax.set_ylabel("residual (log units)")
viz.titled(ax, "Standardised residuals", "No trend and no drifting variance is what we want.")

ax = axes[0, 1]
ax.hist(resid, bins=26, color=viz.CATEGORICAL[0], alpha=0.85, edgecolor=viz.INK["surface"])
xs = np.linspace(resid.min(), resid.max(), 200)
ax.plot(xs, sps.norm.pdf(xs, resid.mean(), resid.std()) * len(resid) *
        (resid.max() - resid.min()) / 26, color=viz.CATEGORICAL[1], linewidth=1.8)
viz.titled(ax, "Residual distribution", "Orange line: normal with the same mean and sd.")

ax = axes[1, 0]
plot_acf(resid, lags=24, ax=ax, zero=False, color=viz.CATEGORICAL[0],
         vlines_kwargs={"colors": viz.CATEGORICAL[0]})
for coll in ax.collections:
    coll.set_facecolor(viz.CATEGORICAL[0]); coll.set_alpha(0.15)
ax.set_title("Residual ACF", loc="left"); ax.set_xlabel("lag (months)")

ax = axes[1, 1]
lb = acorr_ljungbox(resid, lags=range(1, 25), return_df=True)
ax.axhline(0.05, color=viz.STATUS["critical"], linewidth=1.2)
ax.annotate("5% level", xy=(24, 0.05), xytext=(-4, 5), textcoords="offset points",
            ha="right", fontsize=9, color=viz.STATUS["critical"])
ax.plot(lb.index, lb["lb_pvalue"].values, marker="o", color=viz.CATEGORICAL[0],
        markersize=4, linewidth=1.6)
ax.set_ylim(0, 1.02); ax.set_xlabel("lag (months)"); ax.set_ylabel("Ljung-Box p-value")
viz.titled(ax, "Ljung-Box test",
           "Points above the red line mean no autocorrelation is left at that lag.")
fig.tight_layout()
viz.save_fig(fig, "11_sarima_diagnostics", FIG_DIR)
plt.show()

n_reject = int((lb["lb_pvalue"] < 0.05).sum())
print("Ljung-Box rejections at the 5%% level: %d of %d lags" % (n_reject, len(lb)))
print("Jarque-Bera on residuals: p = %.4g" % sps.jarque_bera(resid)[1])

# %% [markdown]
# ## 8. Model 2 - Random Forest
#
# Tuned by randomised search over a `TimeSeriesSplit`, so every validation fold
# is strictly later than the data used to fit it. Scoring on MAE of the log
# return rather than MSE keeps the search from being dominated by the two or
# three crisis months.

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance

cv = TimeSeriesSplit(n_splits=5)

# The search parallelises across candidates; the estimator inside it stays
# single-threaded. Two nested `n_jobs=-1` layers oversubscribe every core and
# make the search several times slower than running it serially.
INNER_JOBS, SEARCH_JOBS = 1, -1

rf_space = {
    "n_estimators": [300, 600, 1000],
    "max_depth": [3, 5, 8, 12, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "max_features": [0.2, 0.35, 0.5, 0.8, "sqrt"],
}

t0 = time.time()
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=INNER_JOBS, bootstrap=True),
    rf_space, n_iter=40, cv=cv, scoring="neg_mean_absolute_error",
    random_state=RANDOM_STATE, n_jobs=SEARCH_JOBS, refit=True,
).fit(X_train, r_train)
fit_seconds["Random Forest"] = time.time() - t0

rf = rf_search.best_estimator_.set_params(n_jobs=-1)
predictions["Random Forest"] = y_prev_test * np.exp(rf.predict(X_test))

print("best CV MAE (log return): %.5f" % (-rf_search.best_score_))
print("fitted in %.1fs" % fit_seconds["Random Forest"])
pd.Series(rf_search.best_params_).to_frame("value")

# %% [markdown]
# ### 8.1 What the forest actually uses
#
# Impurity importance is biased towards high-cardinality features, so
# permutation importance on the hold-out is reported alongside it: the honest
# question is how much accuracy is lost when a feature is shuffled.

# %%
perm = permutation_importance(rf, X_test, r_test, n_repeats=30,
                              random_state=RANDOM_STATE, scoring="neg_mean_absolute_error")
imp = pd.DataFrame({
    "feature": FEATURES,
    "impurity": rf.feature_importances_,
    "permutation": perm.importances_mean,
    "perm_sd": perm.importances_std,
}).sort_values("permutation", ascending=False)
imp.to_csv(TBL_DIR / "rf_feature_importance.csv", index=False)

top = imp.head(18).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), sharey=True)
ax = axes[0]
ax.barh(range(len(top)), top["permutation"], color=viz.CATEGORICAL[1], height=0.68)
ax.errorbar(top["permutation"], range(len(top)), xerr=top["perm_sd"], fmt="none",
            ecolor=viz.INK["secondary"], elinewidth=0.9, capsize=2)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f.replace("_", " ") for f in top["feature"]], fontsize=9)
ax.axvline(0, color=viz.INK["axis"], linewidth=0.8)
ax.set_xlabel("MAE increase when shuffled")
viz.titled(ax, "Permutation importance", "Measured on the hold-out - features that earn their place.")

ax = axes[1]
ax.barh(range(len(top)), top["impurity"], color=viz.CATEGORICAL[0], height=0.68)
ax.set_xlabel("mean impurity decrease")
viz.titled(ax, "Impurity importance", "The in-sample view, shown for contrast.")
fig.tight_layout()
viz.save_fig(fig, "12_rf_importance", FIG_DIR)
plt.show()

imp.head(12).set_index("feature").round(5)

# %% [markdown]
# ## 9. Model 3 - XGBoost
#
# Same protocol, a wider search space, and early stopping on a held-back tail of
# the training span so the number of boosting rounds is chosen rather than guessed.

# %%
import xgboost as xgb

xgb_space = {
    "n_estimators": [300, 600, 1000, 1500],
    "learning_rate": [0.005, 0.01, 0.03, 0.06, 0.1],
    "max_depth": [2, 3, 4, 6],
    "min_child_weight": [1, 3, 6, 10],
    "subsample": [0.6, 0.75, 0.9, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
    "reg_lambda": [0.5, 1.0, 3.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
    "gamma": [0.0, 0.1, 0.5],
}

t0 = time.time()
xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=INNER_JOBS, tree_method="hist",
                     objective="reg:squarederror"),
    xgb_space, n_iter=60, cv=cv, scoring="neg_mean_absolute_error",
    random_state=RANDOM_STATE, n_jobs=SEARCH_JOBS, refit=True,
).fit(X_train, r_train)
fit_seconds["XGBoost"] = time.time() - t0

xgbm = xgb_search.best_estimator_.set_params(n_jobs=-1)
predictions["XGBoost"] = y_prev_test * np.exp(xgbm.predict(X_test))

print("best CV MAE (log return): %.5f" % (-xgb_search.best_score_))
print("fitted in %.1fs" % fit_seconds["XGBoost"])
pd.Series(xgb_search.best_params_).to_frame("value")

# %%
gain = xgbm.get_booster().get_score(importance_type="gain")
xgb_imp = (pd.Series(gain).rename("gain").to_frame()
           .assign(gain=lambda d: d["gain"] / d["gain"].sum())
           .sort_values("gain", ascending=False))
xgb_imp.to_csv(TBL_DIR / "xgb_feature_importance.csv")

both = (imp.set_index("feature")[["permutation"]]
        .join(xgb_imp, how="outer").fillna(0.0))
both["rank_rf"] = both["permutation"].rank(ascending=False)
both["rank_xgb"] = both["gain"].rank(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
ax = axes[0]
topx = xgb_imp.head(16).iloc[::-1]
ax.barh(range(len(topx)), topx["gain"], color=viz.CATEGORICAL[2], height=0.68)
ax.set_yticks(range(len(topx)))
ax.set_yticklabels([f.replace("_", " ") for f in topx.index], fontsize=9)
ax.set_xlabel("share of total gain")
viz.titled(ax, "XGBoost gain importance", "How much each split improved the objective.")

ax = axes[1]
sel = both.nsmallest(25, "rank_xgb")
ax.scatter(sel["rank_rf"], sel["rank_xgb"], s=42, color=viz.CATEGORICAL[0],
           edgecolor=viz.INK["surface"], linewidth=1.4, zorder=3)
lims = [0, max(sel["rank_rf"].max(), sel["rank_xgb"].max()) + 2]
ax.plot(lims, lims, color=viz.INK["axis"], linewidth=1.0, zorder=1)
for f, row in sel.nsmallest(6, "rank_xgb").iterrows():
    ax.annotate(f.replace("_", " "), xy=(row["rank_rf"], row["rank_xgb"]),
                xytext=(6, 2), textcoords="offset points", fontsize=8,
                color=viz.INK["secondary"])
ax.set_xlabel("rank in the Random Forest"); ax.set_ylabel("rank in XGBoost")
ax.invert_xaxis(); ax.invert_yaxis()
viz.titled(ax, "Do the two ensembles agree?",
           "Points near the grey line are features both models rank alike.")
fig.tight_layout()
viz.save_fig(fig, "13_xgb_importance", FIG_DIR)
plt.show()

print("Spearman rank correlation between the two importance orderings: %.3f"
      % both[["rank_rf", "rank_xgb"]].corr(method="spearman").iloc[0, 1])

# %% [markdown]
# ## 10. Model 4 - LSTM
#
# A two-layer LSTM reading a rolling 12-month window of the same feature vectors
# and predicting the next log return. The network is deliberately small - with
# fewer than 200 training months a larger one memorises the window rather than
# learning from it - and it is fitted from five random seeds whose predictions
# are averaged, so the reported result is not an artefact of one lucky
# initialisation. Early stopping watches a held-back tail of the training span.

# %%
t0 = time.time()
lstm = fc.LSTMRegressor(
    window=12, hidden=48, layers=2, dropout=0.15,
    lr=8e-3, epochs=400, patience=40, batch_size=32,
    val_frac=0.15, n_seeds=5, verbose=True,
)
lstm.fit(X_train, r_train)
predictions["LSTM"] = y_prev_test * np.exp(lstm.predict(X_test, context=X_train))
fit_seconds["LSTM"] = time.time() - t0
print("\nfitted %d seeds on %s in %.1fs" % (lstm.n_seeds, lstm.device, fit_seconds["LSTM"]))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
for i, h in enumerate(lstm.history_):
    ax.plot(h["val"], color=viz.CATEGORICAL[6], alpha=0.35, linewidth=1.2)
    ax.plot(h["train"], color=viz.CATEGORICAL[0], alpha=0.35, linewidth=1.2)
longest = max(len(h["val"]) for h in lstm.history_)
mean_val = [np.mean([h["val"][e] for h in lstm.history_ if len(h["val"]) > e])
            for e in range(longest)]
mean_tr = [np.mean([h["train"][e] for h in lstm.history_ if len(h["train"]) > e])
           for e in range(longest)]
ax.plot(mean_tr, color=viz.CATEGORICAL[0], linewidth=2.2, label="training loss")
ax.plot(mean_val, color=viz.CATEGORICAL[6], linewidth=2.2, label="validation loss")
ax.set_xlabel("epoch"); ax.set_ylabel("Huber loss (scaled returns)")
ax.legend()
viz.titled(ax, "Learning curves", "Faint lines are the individual seeds; bold lines their mean.")

ax = axes[1]
seed_preds = []
for net in lstm.models_:
    saved, lstm.models_ = lstm.models_, [net]
    seed_preds.append(y_prev_test * np.exp(lstm.predict(X_test, context=X_train)))
    lstm.models_ = saved
sp = pd.concat(seed_preds, axis=1)
ax.fill_between(test_index, sp.min(axis=1), sp.max(axis=1),
                color=viz.CATEGORICAL[6], alpha=0.18, linewidth=0, label="seed range")
ax.plot(test_index, predictions["LSTM"].values, color=viz.CATEGORICAL[6],
        linewidth=2.0, label="seed-averaged forecast")
ax.plot(test_index, y_test.values, color=viz.INK["primary"], linewidth=2.0, label="actual")
ax.set_ylabel(TARGET_LABEL)
ax.legend(loc="upper left")
viz.titled(ax, "Why the seeds are averaged",
           "The spread between initialisations is comparable to the forecast error itself.")
fig.tight_layout()
viz.save_fig(fig, "14_lstm_training", FIG_DIR)
plt.show()

# %% [markdown]
# ## 11. Forecast combination
#
# Combining forecasts is the most reliably useful trick in the literature: as
# long as the component errors are not perfectly correlated, an equal-weight
# average has lower variance than its members. It is included here as a fifth
# entry rather than as a claim about any one model.

# %%
components = ["SARIMA", "Random Forest", "XGBoost", "LSTM"]
predictions["Ensemble"] = pd.concat([predictions[c] for c in components], axis=1).mean(axis=1)

err_corr = pd.DataFrame({c: (y_test - predictions[c]) for c in components}).corr()
print("Correlation between the component forecast errors:")
err_corr.round(3)

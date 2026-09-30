# %% [markdown]
# ## 12. Results
#
# ### 12.1 The comparison table

# %%
MODEL_ORDER = ["Naive", "Seasonal Naive", "SARIMA", "Random Forest", "XGBoost", "LSTM", "Ensemble"]
ordered = {k: predictions[k] for k in MODEL_ORDER if k in predictions}

results = fc.assemble_results(ordered, y_test, y_train, SEASON)
results = results[["RMSE", "MAE", "MAPE (%)", "sMAPE (%)", "MASE", "Theil U",
                   "R2", "Dir. Acc (%)", "Bias"]]
results.index.name = "Model"
results.to_csv(TBL_DIR / "headline_metrics.csv")

with open(TBL_DIR / "headline_metrics.tex", "w") as fh:
    fh.write(results.round(4).to_latex(
        caption="One-step-ahead forecast accuracy over the %d-month hold-out (%s)."
                % (TEST_MONTHS, SERIES_LABELS.get(PRIMARY_TARGET, PRIMARY_TARGET)),
        label="tab:headline", escape=True))

viz.metric_table_style(results.round(4))

# %%
best_rmse = results["RMSE"].idxmin()
beat_rw = results[results["Theil U"] < 1].index.tolist()

print("Lowest RMSE                     : %s (%.4f)" % (best_rmse, results.loc[best_rmse, "RMSE"]))
print("Lowest MAPE                     : %s (%.2f%%)"
      % (results["MAPE (%)"].idxmin(), results["MAPE (%)"].min()))
print("Best directional accuracy       : %s (%.1f%%)"
      % (results["Dir. Acc (%)"].idxmax(), results["Dir. Acc (%)"].max()))
print("Beat the random walk (U < 1)    : %s"
      % (", ".join(m for m in beat_rw if m != "Naive") or "none"))

# %% [markdown]
# ### 12.2 Forecasts against the outturn

# %%
fig = plt.figure(figsize=(12.5, 7.2))
gs = GridSpec(2, 1, height_ratios=[2.3, 1], hspace=0.48, figure=fig)

ax = fig.add_subplot(gs[0])
context = y.loc[y.index >= test_start - pd.DateOffset(months=18)]
ax.plot(context.index, context.values, color=viz.INK["primary"], linewidth=2.4, label="Actual",
        zorder=6)
ends = []
for name in MODEL_ORDER:
    if name in ("Naive", "Seasonal Naive"):
        continue
    p = predictions[name]
    ax.plot(p.index, p.values, color=viz.MODEL_COLORS[name], linewidth=1.9,
            label=name, alpha=0.95)
    ends.append((p.index[-1], p.iloc[-1], name, viz.MODEL_COLORS[name]))
ax.plot(test_index, predictions["Naive"].values, color=viz.INK["muted"],
        linewidth=1.4, linestyle=(0, (4, 3)), label="Random walk", zorder=2)
# The label goes at the foot of the shaded span so it cannot land under the legend.
viz.shade_span(ax, test_start, test_index[-1])
ax.text(test_start, 0.03, "  hold-out", transform=ax.get_xaxis_transform(),
        color=viz.INK["secondary"], fontsize=9, va="bottom", ha="left")
ax.set_xlim(context.index[0], test_index[-1] + pd.DateOffset(months=11))
ax.set_ylabel(TARGET_LABEL)
ax.legend(loc="upper left", ncol=3)
viz.titled(ax, "One-step-ahead forecasts over the hold-out",
           "Every model is fitted only on data before the shaded window.")
# All four forecasts converge on the final month, so their endpoint labels have
# to be separated explicitly.
viz.label_ends(ax, ends, min_gap=15)

ax = fig.add_subplot(gs[1])
ax.axhline(0, color=viz.INK["axis"], linewidth=0.9)
for name in ["SARIMA", "Random Forest", "XGBoost", "LSTM", "Ensemble"]:
    e = y_test - predictions[name]
    ax.plot(e.index, e.values, color=viz.MODEL_COLORS[name], linewidth=1.6, label=name)
ax.set_ylabel("error (GHS/L)")
viz.titled(ax, "Forecast errors", "Actual minus forecast. Above zero means the model under-predicted.")
viz.save_fig(fig, "15_forecasts_vs_actual", FIG_DIR)
plt.show()

# %% [markdown]
# ### 12.3 Error distributions
#
# The mean error hides whether a model is steadily a little wrong or
# occasionally very wrong. The spread is what a planner actually cares about.

# %%
models_plot = ["Naive", "SARIMA", "Random Forest", "XGBoost", "LSTM", "Ensemble"]
errs = {m: (y_test - predictions[m]).dropna().values for m in models_plot}

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
parts = ax.violinplot([errs[m] for m in models_plot], showmedians=True, widths=0.8)
for i, body in enumerate(parts["bodies"]):
    body.set_facecolor(viz.MODEL_COLORS[models_plot[i]])
    body.set_alpha(0.35); body.set_edgecolor(viz.MODEL_COLORS[models_plot[i]])
for key in ("cmedians", "cbars", "cmins", "cmaxes"):
    if key in parts:
        parts[key].set_color(viz.INK["secondary"]); parts[key].set_linewidth(1.1)
ax.axhline(0, color=viz.INK["axis"], linewidth=0.9)
ax.set_xticks(range(1, len(models_plot) + 1))
ax.set_xticklabels(models_plot, rotation=18, ha="right", fontsize=9)
ax.set_ylabel("error (GHS/L)")
viz.titled(ax, "Distribution of hold-out errors", "Width is how often; height is how badly.")

ax = axes[1]
for m in models_plot:
    a = np.sort(np.abs(errs[m]))
    ax.plot(a, np.arange(1, len(a) + 1) / len(a), color=viz.MODEL_COLORS[m],
            linewidth=1.9, label=m)
ax.set_xlabel("absolute error (GHS/L)"); ax.set_ylabel("share of months at or below")
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")
viz.titled(ax, "Cumulative absolute error",
           "A curve that rises earlier is the better model at every error threshold.")
fig.tight_layout()
viz.save_fig(fig, "16_error_distribution", FIG_DIR)
plt.show()

# %% [markdown]
# ### 12.4 Is the difference real?
#
# A lower RMSE over 24 months can be luck. The Diebold-Mariano test asks whether
# the squared-error sequences differ by more than sampling noise, using the
# Harvey-Leybourne-Newbold correction for the short sample.

# %%
dm_models = ["Naive", "SARIMA", "Random Forest", "XGBoost", "LSTM", "Ensemble"]
dm_stat = pd.DataFrame(np.nan, index=dm_models, columns=dm_models)
dm_p = pd.DataFrame(np.nan, index=dm_models, columns=dm_models)
for a in dm_models:
    for b in dm_models:
        if a == b:
            continue
        s, p = fc.diebold_mariano(y_test - predictions[a], y_test - predictions[b], h=1)
        dm_stat.loc[a, b], dm_p.loc[a, b] = s, p

fig, ax = plt.subplots(figsize=(7.4, 6))
im = ax.imshow(dm_stat.values.astype(float), cmap=viz.DIVERGING, vmin=-3, vmax=3)
ax.set_xticks(range(len(dm_models))); ax.set_xticklabels(dm_models, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(dm_models))); ax.set_yticklabels(dm_models, fontsize=9)
for i, a in enumerate(dm_models):
    for j, b in enumerate(dm_models):
        if i == j:
            ax.text(j, i, "-", ha="center", va="center", color=viz.INK["muted"]); continue
        s, p = dm_stat.iloc[i, j], dm_p.iloc[i, j]
        star = "*" if p < 0.05 else ""
        ax.text(j, i, "%.2f%s" % (s, star), ha="center", va="center", fontsize=8.5,
                color=viz.INK["surface"] if abs(s) > 1.8 else viz.INK["secondary"])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("DM statistic", fontsize=9)
ax.set_xlabel("compared against"); ax.set_ylabel("model")
viz.titled(ax, "Diebold-Mariano tests",
           "Negative (blue) = the row model has smaller errors. * marks p < 0.05.")
fig.tight_layout()
viz.save_fig(fig, "17_diebold_mariano", FIG_DIR)
plt.show()

dm_p.round(3).to_csv(TBL_DIR / "diebold_mariano_pvalues.csv")
sig = [(a, b, dm_stat.loc[a, b], dm_p.loc[a, b])
       for a in dm_models for b in dm_models
       if a != b and pd.notna(dm_p.loc[a, b]) and dm_p.loc[a, b] < 0.05 and dm_stat.loc[a, b] < 0]
if sig:
    print("Significantly better at the 5% level:")
    for a, b, s, p in sig:
        print("   %-14s beats %-14s  DM %.2f, p = %.3f" % (a, b, s, p))
else:
    print("No pair differs significantly at the 5%% level over %d months." % TEST_MONTHS)
    print("With a hold-out this short that is the expected outcome, and it is the")
    print("reason the multi-horizon study below pools errors across many origins.")

# %% [markdown]
# ## 13. How accuracy decays with the horizon
#
# One month ahead, a random walk is a formidable benchmark for any price series.
# The question that matters for planning is what happens further out, where the
# random walk simply freezes at the last observed price. Each origin below is
# re-fitted on data available at that date and rolled forward recursively.

# %%
def multi_horizon_study(frame, spec_, hmax=MAX_HORIZON, n_origins=N_ORIGINS):
    f_all = fc.make_features(frame, spec_).dropna()
    Xa = f_all[fc.feature_columns(f_all)]
    idx = f_all.index
    origins = idx[-(n_origins + hmax):-hmax]
    logy_ = np.log(frame[spec_.target])
    exog_now = [c for c in spec_.exog if c in frame.columns]
    rows = []
    for o in origins:
        tr_ = idx < o
        if tr_.sum() < 5 * SEASON:
            continue
        r_ = f_all.loc[tr_, "r"]
        rf_o = RandomForestRegressor(**{**rf_search.best_params_,
                                        "random_state": RANDOM_STATE, "n_jobs": -1}).fit(Xa[tr_], r_)
        xg_o = xgb.XGBRegressor(**{**xgb_search.best_params_, "random_state": RANDOM_STATE,
                                   "n_jobs": -1, "tree_method": "hist"}).fit(Xa[tr_], r_)
        fut = idx[idx >= o][:hmax]
        hist = frame.loc[frame.index < o]
        exf = frame.loc[fut, exog_now]
        p_rf = fc.recursive_forecast(hist, fut, spec_, lambda z, c: rf_o.predict(z), exf, Xa.columns)
        p_xg = fc.recursive_forecast(hist, fut, spec_, lambda z, c: xg_o.predict(z), exf, Xa.columns)
        try:
            p_sa, _, _ = fc.sarimax_multi_step(logy_, o, SARIMA_ORDER, SARIMA_SEASONAL)
            p_sa = p_sa.iloc[:hmax]
        except Exception:
            p_sa = pd.Series(np.nan, index=fut)
        last = float(frame[spec_.target].loc[frame.index < o].iloc[-1])
        act = frame[spec_.target].loc[fut]
        for k, ts in enumerate(fut, start=1):
            rows.append({"origin": o, "h": k, "actual": act[ts], "Naive": last,
                         "SARIMA": p_sa.get(ts, np.nan),
                         "Random Forest": p_rf[ts], "XGBoost": p_xg[ts]})
    return pd.DataFrame(rows)

t0 = time.time()
mh = multi_horizon_study(data, spec)
mh["Ensemble"] = mh[["SARIMA", "Random Forest", "XGBoost"]].mean(axis=1)
print("%d origins x %d horizons in %.0fs" % (mh["origin"].nunique(), MAX_HORIZON, time.time() - t0))

mh_models = ["Naive", "SARIMA", "Random Forest", "XGBoost", "Ensemble"]
prof = []
for h, g in mh.groupby("h"):
    row = {"h": h}
    base = float(np.sqrt(np.mean((g["Naive"] - g["actual"]) ** 2)))
    for m in mh_models:
        e = g[m] - g["actual"]
        row[m + "|MAPE"] = float(np.mean(np.abs(e / g["actual"])) * 100)
        row[m + "|U"] = float(np.sqrt(np.mean(e ** 2)) / base)
    prof.append(row)
prof = pd.DataFrame(prof).set_index("h")
prof.to_csv(TBL_DIR / "multi_horizon_profile.csv")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
ax = axes[0]
ends = []
for m in mh_models:
    ax.plot(prof.index, prof[m + "|MAPE"], color=viz.MODEL_COLORS[m], linewidth=2.0,
            marker="o", markersize=4, label=m)
    ends.append((prof.index[-1], prof[m + "|MAPE"].iloc[-1], m, viz.MODEL_COLORS[m]))
ax.set_xlabel("forecast horizon (months)"); ax.set_ylabel("MAPE, %")
ax.set_xlim(0.6, MAX_HORIZON + 4.4)
ax.legend(loc="upper left")
viz.titled(ax, "Accuracy decays with the horizon",
           "Pooled over %d rolling origins." % mh["origin"].nunique())
# SARIMA tracks the random walk almost exactly here, so the two labels would
# otherwise print on top of one another.
viz.label_ends(ax, ends, min_gap=15)

ax = axes[1]
ax.axhline(1.0, color=viz.INK["secondary"], linewidth=1.2)
ax.text(0.015, 1.0, "random walk", transform=ax.get_yaxis_transform(),
        ha="left", va="bottom", fontsize=9, color=viz.INK["secondary"],
        bbox={"facecolor": viz.INK["surface"], "edgecolor": "none", "pad": 1.5})
for m in mh_models:
    if m == "Naive":
        continue
    ax.plot(prof.index, prof[m + "|U"], color=viz.MODEL_COLORS[m], linewidth=2.0,
            marker="o", markersize=4, label=m)
ax.set_xlabel("forecast horizon (months)"); ax.set_ylabel("RMSE relative to the random walk")
ax.legend(loc="upper left")
viz.titled(ax, "Who beats the benchmark, and where",
           "Below the grey line is better than freezing the last observed price.")
fig.tight_layout()
viz.save_fig(fig, "18_horizon_profile", FIG_DIR)
plt.show()

prof.round(3)

# %% [markdown]
# ## 14. Replication across every product
#
# A result that holds on one series and nowhere else is a result about that
# series. The whole benchmark is re-run below on every product in the panel,
# with the tuned hyper-parameters carried over and the hold-out scaled to each
# series' length.

# %%
def run_series(col, test_months=TEST_MONTHS, lstm_seeds=3):
    ex = [c for c in product_columns(panel_m.columns, base_product(col)) if c != col]
    ex += [c for c in MARKET_DRIVERS if c != col]
    frame = fc.prepare_series(panel_m, col, ex)
    sp = fc.FeatureSpec(target=col, exog=[c for c in frame.columns if c != col])
    f = fc.make_features(frame, sp).dropna()
    if len(f) < 60:
        return None
    Xc = f[fc.feature_columns(f)]
    h = min(test_months, max(12, int(len(f) * 0.2)))
    ts_ = f.index[-h]
    tr_, te_ = f.index < ts_, f.index >= ts_
    ytr_, yte_, prev_ = f.loc[tr_, "y"], f.loc[te_, "y"], f.loc[te_, "y_lag1"]
    rtr_ = f.loc[tr_, "r"]

    p = {"Naive": prev_.copy()}
    p["Random Forest"] = prev_ * np.exp(
        RandomForestRegressor(**{**rf_search.best_params_, "random_state": RANDOM_STATE,
                                 "n_jobs": -1}).fit(Xc[tr_], rtr_).predict(Xc[te_]))
    p["XGBoost"] = prev_ * np.exp(
        xgb.XGBRegressor(**{**xgb_search.best_params_, "random_state": RANDOM_STATE,
                            "n_jobs": -1, "tree_method": "hist"}
                         ).fit(Xc[tr_], rtr_).predict(Xc[te_]))
    ly = np.log(frame[col])
    g = fc.sarimax_grid_search(ly[ly.index < ts_], p_range=(0, 1, 2), q_range=(0, 1, 2),
                               P_range=(0, 1), D_range=(0, 1), Q_range=(0, 1), season=SEASON)
    b = g.iloc[0]
    p["SARIMA"], _ = fc.sarimax_one_step(ly, ts_, b["order"], b["seasonal_order"])
    lm = fc.LSTMRegressor(window=12, hidden=48, layers=2, epochs=300,
                          n_seeds=lstm_seeds, verbose=False).fit(Xc[tr_], rtr_)
    p["LSTM"] = prev_ * np.exp(lm.predict(Xc[te_], context=Xc[tr_]))
    p["Ensemble"] = pd.concat([p[m] for m in ("SARIMA", "Random Forest", "XGBoost", "LSTM")],
                              axis=1).mean(axis=1)

    tab = fc.assemble_results(p, yte_, ytr_, SEASON)
    tab.insert(0, "series", SERIES_LABELS.get(col, col))
    tab.insert(1, "n_months", len(f))
    tab.insert(2, "test_months", h)
    tab.insert(3, "order", str(b["order"]) + str(b["seasonal_order"]))
    return tab.reset_index().rename(columns={"index": "model"})

t0 = time.time()
rep = []
for s in STUDY_SERIES:
    try:
        out = run_series(s)
        if out is not None:
            rep.append(out)
            best = out.loc[out["Theil U"].idxmin()]
            print("%-32s n=%3d  best: %-14s U=%.3f  MAPE=%5.2f%%"
                  % (SERIES_LABELS.get(s, s), out["n_months"].iloc[0],
                     best["model"], best["Theil U"], best["MAPE (%)"]))
    except Exception as exc:
        print("%-32s skipped (%s)" % (s, exc))
replication = pd.concat(rep, ignore_index=True)
replication.to_csv(TBL_DIR / "replication_all_series.csv", index=False)
print("\ncompleted in %.0fs" % (time.time() - t0))

# %%
heat = replication.pivot_table(index="series", columns="model", values="Theil U")
heat = heat[[m for m in MODEL_ORDER if m in heat.columns]]
heat = heat.loc[heat.mean(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(9.4, 0.52 * len(heat) + 2.4))
im = ax.imshow(heat.values, cmap=viz.DIVERGING, vmin=0.7, vmax=1.3)
ax.set_xticks(range(len(heat.columns))); ax.set_xticklabels(heat.columns, rotation=30, ha="right")
ax.set_yticks(range(len(heat.index))); ax.set_yticklabels(heat.index, fontsize=9)
for i in range(len(heat.index)):
    for j in range(len(heat.columns)):
        v = heat.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8.5,
                    color=viz.INK["surface"] if abs(v - 1) > 0.18 else viz.INK["secondary"])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.75)
cb.set_label("Theil's U2  (blue = beats the random walk)", fontsize=9)
viz.titled(ax, "Every model on every series",
           "Values below 1.00 beat a random walk; 1.00 is the benchmark itself.")
fig.tight_layout()
viz.save_fig(fig, "19_replication_heatmap", FIG_DIR)
plt.show()

# %%
wins = (replication[replication["model"] != "Naive"]
        .loc[lambda d: d.groupby("series")["Theil U"].idxmin(), ["series", "model", "Theil U", "MAPE (%)"]]
        .sort_values("Theil U"))
counts = wins["model"].value_counts().reindex(
    [m for m in MODEL_ORDER if m not in ("Naive", "Seasonal Naive")]).fillna(0)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax = axes[0]
ax.bar(range(len(counts)), counts.values,
       color=[viz.MODEL_COLORS[m] for m in counts.index], width=0.62)
for i, v in enumerate(counts.values):
    ax.text(i, v + 0.06, "%d" % v, ha="center", fontsize=10, color=viz.INK["secondary"])
ax.set_xticks(range(len(counts))); ax.set_xticklabels(counts.index, rotation=20, ha="right")
ax.set_ylabel("series won")
viz.titled(ax, "How often each model wins", "Lowest Theil's U2 on each of the %d series." % len(wins))

ax = axes[1]
# The median, not the mean: one pathological series (Premix, whose price is a
# step function with long flat stretches) sends XGBoost's mean past 2 and would
# otherwise decide the ranking on its own. Bars show the median, whiskers the
# interquartile range across series.
stand = (replication[replication["model"] != "Naive"]
         .groupby("model")["Theil U"].agg(median="median",
                                          q1=lambda s: s.quantile(0.25),
                                          q3=lambda s: s.quantile(0.75)))
stand = stand.reindex([m for m in MODEL_ORDER if m in stand.index])
ax.axhline(1.0, color=viz.INK["secondary"], linewidth=1.2)
ax.bar(range(len(stand)), stand["median"],
       color=[viz.MODEL_COLORS[m] for m in stand.index], width=0.62)
ax.errorbar(range(len(stand)), stand["median"],
            yerr=[stand["median"] - stand["q1"], stand["q3"] - stand["median"]],
            fmt="none", ecolor=viz.INK["secondary"], elinewidth=1.0, capsize=3)
ax.set_xticks(range(len(stand))); ax.set_xticklabels(stand.index, rotation=20, ha="right")
ax.set_ylabel("median Theil's U2 across series")
ax.set_ylim(0, max(1.15, float(stand["q3"].max()) * 1.12))
viz.titled(ax, "Average standing",
           "Median with the interquartile range; the grey line is the benchmark.")
fig.tight_layout()
viz.save_fig(fig, "20_model_standings", FIG_DIR)
plt.show()

wins.set_index("series").round(3)

# %% [markdown]
# ## 15. Ex-ante forecast
#
# The models are now refitted on the **entire** sample and projected 12 months
# past the last observation. The exogenous drivers are held at their final
# observed value over the horizon - the honest assumption when no independent
# forecast of the cedi or of world product prices is available, and one that
# should be stated wherever this projection is quoted.

# %%
future_index = pd.date_range(data.index[-1] + pd.offsets.MonthEnd(1),
                             periods=MAX_HORIZON, freq="ME")

full_feats = fc.make_features(data, spec).dropna()
X_full = full_feats[FEATURES]
r_full = full_feats["r"]

rf_full = RandomForestRegressor(**{**rf_search.best_params_, "random_state": RANDOM_STATE,
                                   "n_jobs": -1}).fit(X_full, r_full)
xgb_full = xgb.XGBRegressor(**{**xgb_search.best_params_, "random_state": RANDOM_STATE,
                               "n_jobs": -1, "tree_method": "hist"}).fit(X_full, r_full)
lstm_full = fc.LSTMRegressor(window=12, hidden=48, layers=2, epochs=400,
                             n_seeds=5, verbose=False).fit(X_full, r_full)

future = pd.DataFrame(index=future_index)
future["Random Forest"] = fc.recursive_forecast(
    data, future_index, spec, lambda z, c: rf_full.predict(z), columns=FEATURES)
future["XGBoost"] = fc.recursive_forecast(
    data, future_index, spec, lambda z, c: xgb_full.predict(z), columns=FEATURES)
future["LSTM"] = fc.recursive_forecast(
    data, future_index, spec, lambda z, c: lstm_full.predict(z, context=c), columns=FEATURES)

sar_full = fc.fit_sarimax(np.log(y), SARIMA_ORDER, SARIMA_SEASONAL)
sf = sar_full.get_forecast(steps=MAX_HORIZON)
future["SARIMA"] = np.exp(sf.predicted_mean).values
sar_ci = np.exp(sf.conf_int(alpha=0.20)); sar_ci.index = future_index
future["Ensemble"] = future[["SARIMA", "Random Forest", "XGBoost", "LSTM"]].mean(axis=1)

# Empirical uncertainty band: the spread of hold-out errors at each horizon,
# which reflects how this ensemble has actually behaved rather than an assumption.
ens_mh = mh.assign(err_pct=lambda d: (d["Ensemble"] - d["actual"]) / d["actual"])
band = ens_mh.groupby("h")["err_pct"].quantile([0.10, 0.90]).unstack()
band.columns = ["q10", "q90"]
future["lower"] = future["Ensemble"] * (1 + band["q10"].reindex(range(1, MAX_HORIZON + 1)).values)
future["upper"] = future["Ensemble"] * (1 + band["q90"].reindex(range(1, MAX_HORIZON + 1)).values)
future.round(3).to_csv(TBL_DIR / "ex_ante_forecast.csv")

fig, ax = plt.subplots(figsize=(12, 5.4))
hist_tail = y.loc[y.index >= y.index[-60]]
ax.plot(hist_tail.index, hist_tail.values, color=viz.INK["primary"], linewidth=2.2, label="Observed")
ax.fill_between(future_index, future["lower"], future["upper"],
                color=viz.CATEGORICAL[4], alpha=0.16, linewidth=0,
                label="80% empirical band")
for m in ["SARIMA", "Random Forest", "XGBoost", "LSTM"]:
    ax.plot(future_index, future[m], color=viz.MODEL_COLORS[m], linewidth=1.5,
            alpha=0.85, label=m)
ax.plot(future_index, future["Ensemble"], color=viz.MODEL_COLORS["Ensemble"],
        linewidth=2.6, label="Ensemble")
ax.plot([hist_tail.index[-1], future_index[0]],
        [hist_tail.iloc[-1], future["Ensemble"].iloc[0]],
        color=viz.MODEL_COLORS["Ensemble"], linewidth=2.6)
ax.axvline(y.index[-1], color=viz.INK["axis"], linewidth=1.0)
ax.annotate("last observation\n%s" % y.index[-1].strftime("%b %Y"),
            xy=(y.index[-1], ax.get_ylim()[0]), xytext=(-8, 12),
            textcoords="offset points", ha="right", fontsize=9, color=viz.INK["secondary"])
viz.label_end(ax, future_index[-1], future["Ensemble"].iloc[-1],
              "%.2f" % future["Ensemble"].iloc[-1], viz.MODEL_COLORS["Ensemble"])
ax.set_xlim(hist_tail.index[0], future_index[-1] + pd.DateOffset(months=5))
ax.set_ylabel(TARGET_LABEL)
ax.legend(loc="upper left", ncol=3)
viz.titled(ax, "Twelve-month ex-ante forecast",
           "Drivers held at their last observed value. The band is the ensemble's own "
           "10th-90th percentile error at each horizon.")
viz.annotate_source(fig, "Source: NPA price workbook; author's models. Not a policy forecast.")
viz.save_fig(fig, "21_ex_ante_forecast", FIG_DIR)
plt.show()

future[["SARIMA", "Random Forest", "XGBoost", "LSTM", "Ensemble", "lower", "upper"]].round(3)

# %% [markdown]
# ## 16. Summary
#
# ### 16.1 Computational cost
#
# Accuracy is not the only axis a thesis should report: a model that takes two
# orders of magnitude longer to fit for a fractional gain is a different
# proposition for a regulator running it monthly.

# %%
cost = pd.DataFrame({
    "fit seconds": pd.Series(fit_seconds),
    "RMSE": results["RMSE"],
    "MAPE (%)": results["MAPE (%)"],
    "Theil U": results["Theil U"],
}).dropna(subset=["fit seconds"])

fig, ax = plt.subplots(figsize=(7.4, 5))
for m in cost.index:
    ax.scatter(cost.loc[m, "fit seconds"], cost.loc[m, "Theil U"], s=150,
               color=viz.MODEL_COLORS[m], edgecolor=viz.INK["surface"], linewidth=2, zorder=3)
    ax.annotate(m, xy=(cost.loc[m, "fit seconds"], cost.loc[m, "Theil U"]),
                xytext=(10, 0), textcoords="offset points", fontsize=10,
                color=viz.MODEL_COLORS[m], va="center")
ax.axhline(1.0, color=viz.INK["secondary"], linewidth=1.2)
ax.annotate("random walk", xy=(ax.get_xlim()[1], 1.0), xytext=(-4, 5),
            textcoords="offset points", ha="right", fontsize=9, color=viz.INK["secondary"])
ax.set_xscale("log")
ax.set_xlabel("fitting time, seconds (log scale)")
ax.set_ylabel("Theil's U2")
ax.set_xlim(right=ax.get_xlim()[1] * 9)   # headroom so the rightmost label fits
viz.titled(ax, "Accuracy against the cost of getting it",
           "Bottom-left is the sweet spot: cheap and accurate.")
viz.save_fig(fig, "22_cost_vs_accuracy", FIG_DIR)
plt.show()

cost.round(3)

# %% [markdown]
# ### 16.2 Findings

# %%
lines = []
lines.append("PRIMARY SERIES : %s" % SERIES_LABELS.get(PRIMARY_TARGET, PRIMARY_TARGET))
lines.append("SAMPLE         : %s - %s (%d months, %d used after feature warm-up)"
             % (y.index[0].strftime("%b %Y"), y.index[-1].strftime("%b %Y"), len(y), len(feats)))
lines.append("HOLD-OUT       : %d months, never trained on" % TEST_MONTHS)
lines.append("")
lines.append("ONE STEP AHEAD")
lines.append("  best RMSE            %-16s %.4f" % (best_rmse, results.loc[best_rmse, "RMSE"]))
lines.append("  best MAPE            %-16s %.2f%%"
             % (results["MAPE (%)"].idxmin(), results["MAPE (%)"].min()))
lines.append("  best direction       %-16s %.1f%%"
             % (results["Dir. Acc (%)"].idxmax(), results["Dir. Acc (%)"].max()))
lines.append("  beat the random walk %s"
             % (", ".join(m for m in beat_rw if m != "Naive") or "none"))
lines.append("")
lines.append("ACROSS HORIZONS 1-%d" % MAX_HORIZON)
for h in (1, 3, 6, MAX_HORIZON):
    if h in prof.index:
        us = {m: prof.loc[h, m + "|U"] for m in mh_models if m != "Naive"}
        bm = min(us, key=us.get)
        lines.append("  h=%-2d  best %-16s U = %.3f   (MAPE %.1f%%)"
                     % (h, bm, us[bm], prof.loc[h, bm + "|MAPE"]))
lines.append("")
lines.append("ACROSS %d PRODUCT SERIES" % replication["series"].nunique())
for m, n in counts.items():
    lines.append("  %-16s won %d series" % (m, int(n)))
lines.append("")
lines.append("SEASONALITY    : strength %.2f - %s"
             % (seas_strength, "material" if seas_strength > 0.4 else "negligible; the selected SARIMA order reflects that"))
lines.append("STATIONARITY   : levels carry a unit root, first differences do not")
lines.append("                 -> d = 1 for SARIMA; returns as the learner target")

report = "\n".join(lines)
print(report)
(TBL_DIR / "summary.txt").write_text(report, encoding="utf-8")

# %% [markdown]
# ### 16.3 Outputs written
#
# Figures are written twice: a 300 dpi PNG for drafts and a vector PDF for the
# typeset thesis.

# %%
figs = sorted(p.name for p in FIG_DIR.glob("*.png"))
tbls = sorted(p.name for p in TBL_DIR.glob("*"))
print("outputs/figures/  (%d)" % len(figs))
for f in figs:
    print("   ", f)
print("\noutputs/tables/   (%d)" % len(tbls))
for t in tbls:
    print("   ", t)

# %% [markdown]
# ---
#
# ### Notes for the write-up
#
# * **The random walk is the benchmark that matters.** Reporting R2 against a
#   trending price level flatters every model; MASE and Theil's U2 are reported
#   here precisely because they cannot be inflated that way. Where a model does
#   not beat the random walk one step ahead, say so - it is a finding about the
#   efficiency of Ghana's pricing windows, not a defect in the model.
#
# * **State the exogenous assumption** wherever the ex-ante forecast appears:
#   the cedi and world product prices are held at their last observed value, so
#   the projection is conditional, not unconditional.
#
# * **Two datasets in the proposal are absent** from this repository - the NPA
#   consumption volumes and the GSS monthly CPI. Section 1.1 records that
#   explicitly. Once obtained they merge on the month index and every cell below
#   re-runs unchanged.
#
# * **Reproducibility.** Every random process is seeded (`RANDOM_STATE = 42`);
#   the LSTM additionally averages five seeds so its result does not depend on
#   one initialisation.

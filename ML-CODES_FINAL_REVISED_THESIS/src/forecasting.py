"""
Forecasting toolkit for the thesis benchmark.

Design decisions that matter for the comparison being fair:

* **Everything is lagged.** ``make_features`` only ever puts information dated
  ``t-1`` or earlier into the row that predicts ``t``, so the machine-learning
  models face the same information set as the one-step-ahead SARIMA forecast.
  Nothing in the design lets a model see the value it is asked to predict.

* **Trees and the LSTM model the log return, not the level.** A tree cannot
  extrapolate beyond the range of its training targets, so asking a random
  forest for the *price* of a series that has tripled since training is asking
  it to fail. Predicting ``r_t = log(y_t) - log(y_{t-1})`` and rebuilding the
  level from the last observed price removes that handicap; SARIMA reaches the
  same place through differencing.

* **Two horizons are reported.** Rolling-origin one-step-ahead (many errors,
  stable metric, refit as the origin moves) is the headline; a single recursive
  multi-step run over the whole hold-out shows trajectory behaviour, which is
  what a reader actually wants from a forecasting chapter.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*date index.*")
warnings.filterwarnings("ignore", message=".*No supported index.*")
try:                                    # statsmodels' own warning classes
    from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning

    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:  # pragma: no cover
    pass

SEED = 42


# ==========================================================================
# Metrics
# ==========================================================================

def _safe(a) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def mape(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true), _safe(y_pred)
    mask = np.abs(y_true) > 1e-9
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true, y_pred = _safe(y_true), _safe(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom > 1e-9
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(y_true, y_pred, y_train, season: int = 12) -> float:
    """Mean absolute scaled error, scaled by the in-sample seasonal naive MAE."""
    y_train = _safe(y_train)
    if len(y_train) <= season:
        season = 1
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale < 1e-12:
        return np.nan
    return float(np.mean(np.abs(_safe(y_true) - _safe(y_pred))) / scale)


def theil_u2(y_true, y_pred) -> float:
    """Theil's U2: forecast RMSE relative to the random-walk RMSE. <1 beats naive."""
    y_true, y_pred = _safe(y_true), _safe(y_pred)
    if len(y_true) < 2:
        return np.nan
    num = np.sqrt(np.mean((y_pred[1:] - y_true[1:]) ** 2))
    den = np.sqrt(np.mean((y_true[:-1] - y_true[1:]) ** 2))
    return float(num / den) if den > 1e-12 else np.nan


def directional_accuracy(y_true, y_pred, y_prev) -> float:
    """Share of periods where the sign of the predicted change is right."""
    y_true, y_pred, y_prev = _safe(y_true), _safe(y_pred), _safe(y_prev)
    actual = np.sign(y_true - y_prev)
    pred = np.sign(y_pred - y_prev)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return float((actual[mask] == pred[mask]).mean() * 100)


def regression_metrics(y_true, y_pred, y_train=None, y_prev=None, season: int = 12) -> Dict[str, float]:
    y_true, y_pred = _safe(y_true), _safe(y_pred)
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    out = {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE (%)": mape(y_true, y_pred),
        "sMAPE (%)": smape(y_true, y_pred),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan,
        "Theil U": theil_u2(y_true, y_pred),
        "Bias": float(np.mean(err)),
    }
    out["MASE"] = mase(y_true, y_pred, y_train, season) if y_train is not None else np.nan
    out["Dir. Acc (%)"] = (
        directional_accuracy(y_true, y_pred, y_prev) if y_prev is not None else np.nan
    )
    return out


def diebold_mariano(e1, e2, h: int = 1, power: int = 2) -> Tuple[float, float]:
    """
    Harvey-Leybourne-Newbold small-sample corrected Diebold-Mariano test.

    Returns ``(DM statistic, two-sided p-value)``. A negative statistic with a
    small p-value says model 1's errors are significantly smaller than model 2's.
    """
    e1, e2 = _safe(e1), _safe(e2)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    if n < 3:
        return np.nan, np.nan
    d_bar = d.mean()
    gamma = [np.sum((d[k:] - d_bar) * (d[:n - k] - d_bar)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = d_bar / np.sqrt(var_d)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= correction
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


# ==========================================================================
# Feature engineering
# ==========================================================================

@dataclass
class FeatureSpec:
    """
    ``exog`` are market variables that are genuinely unknown at forecast time
    (world product prices, the cedi/dollar rate) -- they enter lagged only.

    ``known_exog`` are the administratively set components of the NPA pricing
    formula (specific taxes, levies and the regulated margins under LI 2186).
    Those are gazetted before the pricing window opens, so their value for the
    month being forecast is genuinely known in advance and may enter
    contemporaneously -- the same status a holiday calendar has in a demand
    model. Leave the list empty for a strictly autoregressive benchmark.
    """

    target: str
    exog: List[str] = field(default_factory=list)
    known_exog: List[str] = field(default_factory=list)
    lags: Sequence[int] = (1, 2, 3, 6, 12)
    roll_windows: Sequence[int] = (3, 6, 12)
    exog_lags: Sequence[int] = (1, 2, 3)
    add_calendar: bool = True
    add_fourier: int = 2


def make_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """
    Build a supervised frame indexed by the month being predicted.

    Columns:
      ``y``      the level at t (the thing being forecast)
      ``y_lag1`` the level at t-1 (needed to rebuild a level from a return)
      ``r``      the log return at t (the modelling target for trees/LSTM)
      the rest   features, every one of them dated t-1 or earlier
    """
    d = df.copy()
    y = d[spec.target].astype(float)
    logy = np.log(y.where(y > 0))

    out = pd.DataFrame(index=d.index)
    out["y"] = y
    out["y_lag1"] = y.shift(1)
    out["r"] = logy.diff()

    # --- autoregressive structure of the return -------------------------
    for L in spec.lags:
        out["r_lag%d" % L] = out["r"].shift(L)
    for w in spec.roll_windows:
        base = out["r"].shift(1)
        out["r_rollmean%d" % w] = base.rolling(w).mean()
        out["r_rollstd%d" % w] = base.rolling(w).std()
    out["r_rollmin6"] = out["r"].shift(1).rolling(6).min()
    out["r_rollmax6"] = out["r"].shift(1).rolling(6).max()

    # --- level context, expressed as ratios so it stays scale-free ------
    lag1 = y.shift(1)
    for w in (3, 6, 12):
        out["lvl_ratio_ma%d" % w] = lag1 / lag1.rolling(w).mean()
    out["lvl_ratio_12m"] = lag1 / y.shift(13)
    out["log_y_lag1"] = np.log(lag1.where(lag1 > 0))

    # --- market drivers: lagged only, never contemporaneous -------------
    for col in spec.exog:
        if col not in d.columns:
            continue
        s = d[col].astype(float)
        # Tax columns go negative (subsidy/cross-subsidy windows), so a log
        # return is undefined there; fall back to a plain difference.
        growth = np.log(s).diff() if (s.dropna() > 0).all() else s.diff()
        for L in spec.exog_lags:
            out["%s_lag%d" % (col, L)] = s.shift(L)
            out["%s_chg%d" % (col, L)] = growth.shift(L)
        out["%s_roll3" % col] = s.shift(1).rolling(3).mean()
        ma6 = s.shift(1).rolling(6).mean()
        out["%s_rel_ma6" % col] = s.shift(1) / ma6.where(ma6.abs() > 1e-9)

    # --- policy components: known before the window opens ---------------
    for col in spec.known_exog:
        if col not in d.columns:
            continue
        s = d[col].astype(float)
        out["%s_now" % col] = s                     # contemporaneous by design
        out["%s_chg_now" % col] = s.diff()
        out["%s_share_lag1" % col] = s / lag1.where(lag1.abs() > 1e-9)

    # --- calendar --------------------------------------------------------
    if spec.add_calendar:
        month = d.index.month
        out["month"] = month
        out["time_idx"] = np.arange(len(d))
        for k in range(1, spec.add_fourier + 1):
            out["sin%d" % k] = np.sin(2 * np.pi * k * month / 12)
            out["cos%d" % k] = np.cos(2 * np.pi * k * month / 12)

    # A column that is entirely missing (a component the NPA never published
    # for this product) would otherwise wipe out every row on dropna().
    # A column that is entirely missing (a component the NPA never published for
    # this product) would otherwise wipe out every row on dropna(). Constant
    # columns are deliberately *kept*: dropping them here would make the column
    # set depend on the data window, and the recursive forecaster extends that
    # window one month at a time.
    out = out.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in out.columns if c in ("y", "y_lag1", "r") or out[c].notna().any()]
    return out[keep]


def feature_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c not in ("y", "y_lag1", "r")]


def prepare_series(monthly: pd.DataFrame, target: str, exog: Sequence[str],
                   freq: str = "ME") -> pd.DataFrame:
    """
    Slice the panel down to one target plus its exogenous drivers.

    Trims to the target's own span, fills gaps in the drivers (a driver that
    starts later than the target is back-filled rather than dropped, so a short
    exogenous series never truncates the target), and stamps an explicit
    monthly frequency -- statsmodels refuses to forecast without one.
    """
    exog = list(dict.fromkeys(c for c in exog if c in monthly.columns and c != target))
    df = monthly[[target] + exog].copy()
    df = df.loc[df[target].first_valid_index(): df[target].last_valid_index()]
    df = df.asfreq(freq)

    # A handful of months carry no NPA pricing window. Interpolating them keeps
    # the index regular (statsmodels requires that) and is harmless at this
    # density; the count is recorded so the thesis can state it.
    gaps = int(df[target].isna().sum())
    df[target] = df[target].interpolate(method="time", limit_direction="both")
    if exog:
        df[exog] = df[exog].interpolate(method="time", limit_direction="both").ffill().bfill()
        df = df.drop(columns=[c for c in exog if df[c].isna().all()])
    df.attrs["imputed_months"] = gaps
    df.attrs["target"] = target
    return df


# ==========================================================================
# Baselines
# ==========================================================================

def seasonal_naive_forecast(y: pd.Series, test_index: pd.Index, season: int = 12) -> pd.Series:
    """y_hat(t) = y(t - season): the value one full seasonal cycle earlier."""
    return y.shift(season).reindex(test_index)


def naive_forecast(y: pd.Series, test_index: pd.Index) -> pd.Series:
    """Random walk: the one-step-ahead forecast is the last observed value."""
    return y.shift(1).reindex(test_index)


def drift_forecast(y: pd.Series, train_end, test_index: pd.Index) -> pd.Series:
    """Random walk with drift, projected recursively over the hold-out."""
    train = y.loc[:train_end]
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    steps = np.arange(1, len(test_index) + 1)
    return pd.Series(train.iloc[-1] + slope * steps, index=test_index)


# ==========================================================================
# SARIMA / SARIMAX
# ==========================================================================

def sarimax_grid_search(
    endog: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    p_range=(0, 1, 2),
    d_range=(1,),
    q_range=(0, 1, 2),
    P_range=(0, 1),
    D_range=(0, 1),
    Q_range=(0, 1),
    season: int = 12,
    trend: Optional[str] = None,
    max_models: Optional[int] = None,
) -> pd.DataFrame:
    """Exhaustive AICc search over a SARIMA(X) order grid. Returns a ranked table."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    rows = []
    n = 0
    for p in p_range:
        for d in d_range:
            for q in q_range:
                for P in P_range:
                    for D in D_range:
                        for Q in Q_range:
                            if p == q == P == Q == 0 and d == 0 and D == 0:
                                continue
                            if max_models and n >= max_models:
                                break
                            try:
                                res = SARIMAX(
                                    endog,
                                    exog=exog,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, season),
                                    trend=trend,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False, maxiter=200)
                                k = res.df_model + 1
                                nobs = res.nobs
                                aicc = (
                                    res.aic + (2 * k * (k + 1)) / max(nobs - k - 1, 1)
                                    if nobs > k + 1 else np.inf
                                )
                                rows.append(
                                    {
                                        "order": (p, d, q),
                                        "seasonal_order": (P, D, Q, season),
                                        "AIC": res.aic,
                                        "AICc": aicc,
                                        "BIC": res.bic,
                                        "loglik": res.llf,
                                        "converged": bool(res.mle_retvals.get("converged", True))
                                        if hasattr(res, "mle_retvals") else True,
                                    }
                                )
                            except Exception:
                                pass
                            n += 1
    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError("SARIMAX grid search produced no fitted models")
    grid["has_seasonal"] = [bool(so[0] or so[1] or so[2]) for so in grid["seasonal_order"]]
    return grid.sort_values("AICc").reset_index(drop=True)


def fit_sarimax(endog: pd.Series, order, seasonal_order, exog=None, trend=None):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(
        endog, exog=exog, order=tuple(order), seasonal_order=tuple(seasonal_order),
        trend=trend, enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)


def sarimax_one_step(logy: pd.Series, test_start, order, seasonal_order,
                     exog: Optional[pd.DataFrame] = None, trend=None):
    """
    Fit on the training span, then *filter* (not refit) the hold-out so each
    hold-out prediction conditions only on observations strictly before it.
    Returns ``(level forecast, fitted training results)``.
    """
    tr = logy.index < test_start
    exog_tr = exog[tr] if exog is not None else None
    exog_te = exog[~tr] if exog is not None else None
    res = fit_sarimax(logy[tr], order, seasonal_order, exog=exog_tr, trend=trend)
    ext = res.append(logy[~tr], exog=exog_te, refit=False)
    osa = ext.get_prediction(start=logy.index[~tr][0]).predicted_mean
    osa.index = logy.index[~tr]
    return np.exp(osa), res


def sarimax_multi_step(logy: pd.Series, test_start, order, seasonal_order,
                       exog: Optional[pd.DataFrame] = None, trend=None, alpha: float = 0.20):
    """Fit on train and forecast the whole hold-out recursively, with intervals."""
    tr = logy.index < test_start
    h = int((~tr).sum())
    exog_tr = exog[tr] if exog is not None else None
    exog_te = exog[~tr] if exog is not None else None
    res = fit_sarimax(logy[tr], order, seasonal_order, exog=exog_tr, trend=trend)
    fc_obj = res.get_forecast(steps=h, exog=exog_te)
    mean = np.exp(fc_obj.predicted_mean)
    ci = np.exp(fc_obj.conf_int(alpha=alpha))
    mean.index = logy.index[~tr]
    ci.index = logy.index[~tr]
    ci.columns = ["lower", "upper"]
    return mean, ci, res


# ==========================================================================
# Recursive multi-step forecasting for return-space models
# ==========================================================================

def recursive_forecast(
    history: pd.DataFrame,
    future_index: pd.Index,
    spec: FeatureSpec,
    predict_fn,
    exog_future: Optional[pd.DataFrame] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    Roll a return-space model forward step by step.

    At each step the predicted return rebuilds the price level, the level is
    appended to the working history, features are recomputed from that history,
    and the next step is predicted. Exogenous columns over the horizon come from
    ``exog_future`` when supplied and are otherwise held at their last value.
    For ex-ante evaluation, only pass ``exog_future`` when those values would
    genuinely be known at the forecast origin; otherwise leave it as None to
    use a last-observation-carried-forward scenario and avoid future leakage.

    ``predict_fn`` is called as ``predict_fn(x_row, context)`` where ``context``
    is every feature row up to but excluding ``x_row``. A tree model ignores the
    second argument; a sequence model needs it, because its input window must
    slide over the forecasts already made rather than over a frozen history.
    """
    work = history.copy().astype(float)
    preds = []
    for ts in future_index:
        row = {}
        for c in work.columns:
            if c == spec.target:
                row[c] = np.nan
            elif exog_future is not None and c in exog_future.columns and ts in exog_future.index:
                row[c] = float(np.asarray(exog_future.loc[ts, c]).ravel()[0])
            else:
                row[c] = float(work[c].iloc[-1])
        work = pd.concat([work, pd.DataFrame([row], index=[ts], columns=work.columns)])

        feats = make_features(work, spec)
        # Align to the exact column set the model was fitted on: extending the
        # history can otherwise change which columns survive, and a silently
        # reordered matrix produces silently wrong forecasts.
        cols = list(columns) if columns is not None else feature_columns(feats)
        aligned = feats.reindex(columns=cols)
        x = aligned.loc[[ts]]
        ctx = aligned.loc[aligned.index < ts].dropna()
        r_hat = float(np.asarray(predict_fn(x, ctx)).ravel()[0])
        last_level = float(work[spec.target].iloc[-2])
        y_hat = last_level * float(np.exp(r_hat))
        work.loc[ts, spec.target] = y_hat
        preds.append(y_hat)
    return pd.Series(preds, index=future_index, name="forecast")


# ==========================================================================
# LSTM (PyTorch)
# ==========================================================================

class LSTMRegressor:
    """
    A small stacked LSTM over a window of past feature vectors, predicting the
    next log return. Deliberately modest in size: with ~200 monthly observations
    anything larger memorises the training window.

    ``n_seeds`` fits the same architecture several times and averages the
    predictions -- a seed ensemble, which removes most of the run-to-run
    variance that otherwise makes a deep-learning baseline look arbitrary.
    """

    def __init__(
        self,
        window: int = 12,
        hidden: int = 48,
        layers: int = 2,
        dropout: float = 0.15,
        lr: float = 8e-3,
        epochs: int = 400,
        patience: int = 40,
        batch_size: int = 32,
        val_frac: float = 0.15,
        n_seeds: int = 5,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        self.window = window
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.n_seeds = n_seeds
        self.verbose = verbose
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.models_: List = []
        self.history_: List[Dict[str, List[float]]] = []
        self.feature_names_: List[str] = []

    # -- internals --------------------------------------------------------

    def _build(self, n_features: int):
        torch, nn = self.torch, self.torch.nn

        class Net(nn.Module):
            def __init__(self, n_in, hidden, layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_in, hidden, num_layers=layers, batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1)
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net(n_features, self.hidden, self.layers, self.dropout).to(self.device)

    def _windows(self, X: np.ndarray, y: np.ndarray):
        xs, ys = [], []
        for i in range(self.window, len(X) + 1):
            xs.append(X[i - self.window:i])
            ys.append(y[i - 1])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    # -- public API -------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series):
        torch = self.torch
        self.feature_names_ = list(X.columns)
        Xv = X.to_numpy(dtype=float)
        yv = np.asarray(y, dtype=float)

        self.x_mu_, self.x_sd_ = Xv.mean(0), Xv.std(0) + 1e-8
        self.y_mu_, self.y_sd_ = yv.mean(), yv.std() + 1e-8
        Xs = (Xv - self.x_mu_) / self.x_sd_
        ys = (yv - self.y_mu_) / self.y_sd_

        Xw, yw = self._windows(Xs, ys)
        n_val = max(4, int(len(Xw) * self.val_frac))
        Xtr, ytr = Xw[:-n_val], yw[:-n_val]
        Xva, yva = Xw[-n_val:], yw[-n_val:]

        self.models_, self.history_ = [], []
        for seed in range(self.n_seeds):
            torch.manual_seed(SEED + seed)
            np.random.seed(SEED + seed)
            net = self._build(Xw.shape[2])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=12)
            lossf = torch.nn.SmoothL1Loss(beta=0.6)

            Xtr_t = torch.tensor(Xtr, device=self.device)
            ytr_t = torch.tensor(ytr, device=self.device)
            Xva_t = torch.tensor(Xva, device=self.device)
            yva_t = torch.tensor(yva, device=self.device)

            best, best_state, wait = np.inf, None, 0
            hist = {"train": [], "val": []}
            n = len(Xtr_t)
            for ep in range(self.epochs):
                net.train()
                perm = torch.randperm(n, device=self.device)
                tot = 0.0
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    opt.zero_grad()
                    loss = lossf(net(Xtr_t[idx]), ytr_t[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                    tot += float(loss) * len(idx)
                net.eval()
                with torch.no_grad():
                    vl = float(lossf(net(Xva_t), yva_t))
                sched.step(vl)
                hist["train"].append(tot / n)
                hist["val"].append(vl)
                if vl < best - 1e-6:
                    best, wait = vl, 0
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.models_.append(net)
            self.history_.append(hist)
            if self.verbose:
                print("  seed %d: best val %.5f after %d epochs" % (seed, best, len(hist["val"])))

        # Keep the tail of the training window so a single-row predict has context.
        self.context_ = Xs[-(self.window - 1):] if self.window > 1 else np.zeros((0, Xs.shape[1]))
        return self

    def _predict_scaled(self, Xw: np.ndarray) -> np.ndarray:
        torch = self.torch
        t = torch.tensor(Xw, device=self.device)
        with torch.no_grad():
            preds = [m(t).cpu().numpy() for m in self.models_]
        return np.mean(preds, axis=0)

    def predict(self, X: pd.DataFrame, context: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Predict for every row of ``X``. ``context`` supplies the rows immediately
        preceding ``X`` so the first prediction has a full window; when omitted
        the tail of the training set is used.
        """
        Xv = X[self.feature_names_].to_numpy(dtype=float)
        Xs = (Xv - self.x_mu_) / self.x_sd_
        if context is not None and len(context):
            ctx = (context[self.feature_names_].to_numpy(dtype=float) - self.x_mu_) / self.x_sd_
            ctx = ctx[-(self.window - 1):] if self.window > 1 else ctx[:0]
        else:
            ctx = self.context_
        full = np.vstack([ctx, Xs]).astype(np.float32)
        windows = np.asarray(
            [full[i - self.window:i] for i in range(self.window, len(full) + 1)], dtype=np.float32
        )
        out = self._predict_scaled(windows)
        return out * self.y_sd_ + self.y_mu_


# ==========================================================================
# Evaluation protocol
# ==========================================================================

def rolling_origin_splits(index: pd.Index, test_start, refit_every: int = 6):
    """Yield (train_end, origin) pairs for expanding-window one-step-ahead evaluation."""
    test_idx = index[index >= test_start]
    for k, ts in enumerate(test_idx):
        pos = index.get_loc(ts)
        yield index[pos - 1], ts, (k % refit_every == 0)


def assemble_results(preds: Dict[str, pd.Series], actual: pd.Series,
                     y_train: pd.Series, season: int = 12) -> pd.DataFrame:
    """Build the metric table for a dict of ``{model name: forecast series}``."""
    prev = actual.shift(1)
    rows = {}
    for name, p in preds.items():
        p = p.reindex(actual.index)
        mask = p.notna() & actual.notna()
        rows[name] = regression_metrics(
            actual[mask], p[mask], y_train=y_train,
            y_prev=prev[mask], season=season,
        )
    return pd.DataFrame(rows).T

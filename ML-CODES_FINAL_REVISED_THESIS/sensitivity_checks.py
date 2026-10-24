import sys, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import npa_etl, forecasting as fc
ROOT = Path(__file__).resolve().parent
panel=npa_etl.build_panel(ROOT/'data')
monthly=panel['monthly'].copy(); windows=panel['wide'].copy()
for p in ['ATK','MGO_FOREIGN','AGO_MINES_EXP']:
 col=p+'_price'
 if col in windows.columns:
  converted_col=p.replace('_EXP','')+'_ghs'
  windows[converted_col]=windows[col]*windows['fx_ghs_usd']
  monthly[converted_col]=windows[converted_col].resample('ME').mean().reindex(monthly.index)
target='AGO_MINES_ghs'
market=['fx_ghs_usd','ATK_price','MGO_FOREIGN_price','AGO_MINES_EXP_price','RFO_price']
own=[c for c in npa_etl.product_columns(monthly.columns,'AGO_MINES_EXP') if c!=target]
exog=list(dict.fromkeys(own+[c for c in market if c!=target]))
# Reproduce primary frame and separate observed-only frame
base=fc.prepare_series(monthly,target,exog)
# Base interpolated frame is the analysis series.
# Observed-only frame keeps missing target months missing, while exogenous series use same fill rules.
raw=monthly[[target]+[c for c in exog if c in monthly.columns]].copy()
raw=raw.loc[raw[target].first_valid_index():raw[target].last_valid_index()].asfreq('ME')
exo=[c for c in raw.columns if c!=target]
if exo: raw[exo]=raw[exo].interpolate(method='time',limit_direction='both').ffill().bfill(); raw=raw.drop(columns=[c for c in exo if raw[c].isna().all()])
# use same exogenous spec, lags, calendar
spec=fc.FeatureSpec(target=target,exog=[c for c in base.columns if c!=target],lags=(1,2,3,6,12),roll_windows=(3,6,12),exog_lags=(1,2,3),add_calendar=True,add_fourier=2)
params={'n_estimators':300,'min_samples_split':2,'min_samples_leaf':8,'max_features':0.8,'max_depth':12,'random_state':42,'n_jobs':-1,'bootstrap':True}
def fit_eval(frame,label,feature_mode='full'):
 f=fc.make_features(frame,spec).dropna()
 # final 24 months based on the common calendar test period, not last 24 surviving rows
 test_start=pd.Timestamp('2024-09-30')
 train=f.index<test_start; test=f.index>=test_start
 feats=fc.feature_columns(f)
 if feature_mode=='own':
  feats=[c for c in feats if c.startswith(('r_lag','r_roll','lvl_ratio','log_y_lag1','month','time_idx','sin','cos'))]
 Xtr=f.loc[train,feats]; Xte=f.loc[test,feats]; rtr=f.loc[train,'r']; rtest=f.loc[test,'r']
 model=RandomForestRegressor(**params).fit(Xtr,rtr)
 pred=f.loc[test,'y_lag1']*np.exp(model.predict(Xte))
 actual=f.loc[test,'y']
 mask=pred.notna()&actual.notna()
 err=pred[mask]-actual[mask]
 return {'variant':label,'feature_mode':feature_mode,'n_feature_rows':len(f),'n_train':int(train.sum()),'n_test':int(test.sum()),'RMSE':float(np.sqrt(np.mean(err**2))),'MAE':float(np.mean(np.abs(err))),'MAPE_pct':float(np.mean(np.abs(err/actual[mask]))*100),'features':len(feats)}
results=[]
for frame,label in [(base,'Interpolated target'),(raw,'Observed-only target')]:
 for mode in ['full','own']:
  t=time.time(); results.append(fit_eval(frame,label,mode)); print(results[-1], 'seconds',round(time.time()-t,1))
# conventional MASE scaling: in-sample one-step naive MAE, i.e. mean absolute first difference on training series
f=fc.make_features(base,spec).dropna(); test_start=pd.Timestamp('2024-09-30'); ytr=f.loc[f.index<test_start,'y']
scale=float(ytr.diff().abs().dropna().mean())
# test metrics from RF full interpolated
r=[x for x in results if x['variant']=='Interpolated target' and x['feature_mode']=='full'][0]
print('in_sample_naive_scale_mae',scale,'MASE_if_RF_MAE_scaled_by_nonseasonal_naive',r['MAE']/scale)
pd.DataFrame(results).to_csv(str(ROOT / 'outputs' / 'tables' / 'sensitivity_results.csv'),index=False)
# paired DM and bootstrap for interpolated full-feature RF vs random walk
f=fc.make_features(base,spec).dropna(); test_start=pd.Timestamp('2024-09-30'); train=f.index<test_start; test=f.index>=test_start
features=fc.feature_columns(f); Xtr=f.loc[train,features]; Xte=f.loc[test,features]; rtr=f.loc[train,'r']
rf=RandomForestRegressor(**params).fit(Xtr,rtr)
ytrue=f.loc[test,'y'].to_numpy(); rf_pred=(f.loc[test,'y_lag1']*np.exp(rf.predict(Xte))).to_numpy(); naive_pred=f.loc[test,'y_lag1'].to_numpy()
e_rf=rf_pred-ytrue; e_nv=naive_pred-ytrue
stat,pval=fc.diebold_mariano(e_rf,e_nv,h=1,power=2)
# bootstrap paired rows with replacement, 20,000 replicates
rng=np.random.default_rng(20261024); n=len(ytrue); idxs=rng.integers(0,n,size=(20000,n))
rmse_d=[]; mae_d=[]
for ix in idxs:
 rmse_d.append(np.sqrt(np.mean(e_rf[ix]**2))-np.sqrt(np.mean(e_nv[ix]**2)))
 mae_d.append(np.mean(np.abs(e_rf[ix]))-np.mean(np.abs(e_nv[ix])))
print('RF_vs_naive_DM_stat',stat,'p',pval)
print('RMSE_difference_RF_minus_naive',np.sqrt(np.mean(e_rf**2))-np.sqrt(np.mean(e_nv**2)),'bootstrap_CI',np.quantile(rmse_d,[.025,.975]).tolist())
print('MAE_difference_RF_minus_naive',np.mean(np.abs(e_rf))-np.mean(np.abs(e_nv)),'bootstrap_CI',np.quantile(mae_d,[.025,.975]).tolist())
print('absolute_error_means',np.mean(np.abs(e_rf)),np.mean(np.abs(e_nv)))
# Largest RF absolute forecast errors with dates, plus naive comparison
errtab=pd.DataFrame({'actual':ytrue,'rf_forecast':rf_pred,'naive_forecast':naive_pred},index=f.loc[test].index)
errtab['rf_error']=errtab['actual']-errtab['rf_forecast']; errtab['naive_error']=errtab['actual']-errtab['naive_forecast']
errtab['abs_rf_error']=errtab['rf_error'].abs(); errtab['abs_naive_error']=errtab['naive_error'].abs()
print('LARGEST_RF_ERRORS\n',errtab.sort_values('abs_rf_error',ascending=False).head(5).round(3).to_string())
print('80 pct nominal interval coverage not assessed: current band derived from rolling-origin pooled errors; calibration separate.')

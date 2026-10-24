import sys, warnings
from pathlib import Path
import numpy as np,pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import npa_etl,forecasting as fc
ROOT = Path(__file__).resolve().parent
panel=npa_etl.build_panel(ROOT/'data'); monthly=panel['monthly'].copy(); windows=panel['wide'].copy()
for product in ['ATK','MGO_FOREIGN','AGO_MINES_EXP']:
 c=product+'_price'; cc=product.replace('_EXP','')+'_ghs'
 if c in windows: windows[cc]=windows[c]*windows['fx_ghs_usd']; monthly[cc]=windows[cc].resample('ME').mean().reindex(monthly.index)
target='AGO_MINES_ghs'; own=[c for c in npa_etl.product_columns(monthly.columns,'AGO_MINES_EXP') if c!=target]
market=['fx_ghs_usd','ATK_price','MGO_FOREIGN_price','AGO_MINES_EXP_price','RFO_price']
ex=list(dict.fromkeys(own+[c for c in market if c!=target]))
data=fc.prepare_series(monthly,target,ex); logy=np.log(data[target]); ts=pd.Timestamp('2024-09-30'); tr=logy.index<ts; te=~tr; idx=logy.index[te]
ytrue=data[target].loc[idx]
rows=[]
def metrics(name,p):
 p=pd.Series(np.asarray(p),index=idx); e=p-ytrue
 rows.append({'Model':name,'RMSE':float(np.sqrt(np.mean(e**2))),'MAE':float(np.mean(np.abs(e))),'MAPE_pct':float(np.mean(np.abs(e/ytrue))*100)})
 print(rows[-1])
# Random walk with drift: constant drift in differenced log price
res=SARIMAX(logy[tr],order=(0,1,0),trend='t',enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=300)
extres=res.append(logy[te],refit=False)
pred=np.exp(extres.get_prediction(start=idx[0],end=idx[-1]).predicted_mean); pred.index=idx
metrics('Random walk with drift (ARIMA(0,1,0)+drift)',pred)
# ARIMAX using lagged external drivers; all values are known by t-1
cols=[c for c in ['fx_ghs_usd','ATK_price','MGO_FOREIGN_price','AGO_MINES_EXP_price','RFO_price'] if c in data]
exog=data[cols].shift(1)
# ensure no missing; prepare_series filled series, first shifted row becomes NA and is removed consistently
mask=exog.notna().all(axis=1)
ly=logy[mask]; xx=exog[mask]; tr2=ly.index<ts; te2=ly.index>=ts; idx2=ly.index[te2]
resx=SARIMAX(ly[tr2],exog=xx[tr2],order=(1,1,0),trend='n',enforce_stationarity=False,enforce_invertibility=False).fit(disp=False,maxiter=300)
extx=resx.append(ly[te2],exog=xx[te2],refit=False)
predx=np.exp(extx.get_prediction(start=idx2[0],end=idx2[-1],exog=xx[te2]).predicted_mean); predx.index=idx2
# align actual to idx2
act2=data[target].loc[idx2]; err=predx-act2
rows.append({'Model':'ARIMAX(1,1,0) with lagged market drivers','RMSE':float(np.sqrt(np.mean(err**2))),'MAE':float(np.mean(np.abs(err))),'MAPE_pct':float(np.mean(np.abs(err/act2))*100)})
print(rows[-1])
# ETS model state-space on log prices; append observed holdout without refit, then get one-step prediction.
try:
 ets=ETSModel(logy[tr],error='add',trend='add',damped_trend=True,seasonal=None).fit(disp=False,maxiter=1000)
 etsex=ets.append(logy[te],refit=False)
 pe=np.exp(etsex.get_prediction(start=idx[0],end=idx[-1]).predicted_mean); pe.index=idx
 metrics('ETS additive error, damped additive trend',pe)
except Exception as e:
 print('ETS ERROR',repr(e))
pd.DataFrame(rows).to_csv(str(ROOT / 'outputs' / 'tables' / 'classical_benchmark_results.csv'),index=False)
# Expanding one-step ETS with smoothing parameters selected on the training window and held fixed.
from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
 basefit=ExponentialSmoothing(logy[tr],trend='add',damped_trend=True,seasonal=None,initialization_method='estimated').fit(optimized=True,use_brute=True)
 alpha=basefit.params.get('smoothing_level',0.2); beta=basefit.params.get('smoothing_trend',0.05); phi=basefit.params.get('damping_trend',0.98)
 ets_preds=[]
 for j,ts0 in enumerate(idx):
  hist=logy.loc[logy.index<ts0]
  m=ExponentialSmoothing(hist,trend='add',damped_trend=True,seasonal=None,initialization_method='estimated')
  fit=m.fit(smoothing_level=alpha,smoothing_trend=beta,damping_trend=phi,optimized=False,remove_bias=False)
  ets_preds.append(float(np.exp(fit.forecast(1).iloc[0])))
 metrics('ETS additive error, damped additive trend (expanding one-step)',ets_preds)
 print('ETS fixed smoothing parameters',alpha,beta,phi)
except Exception as e:
 print('ETS ERROR 2',repr(e))
pd.DataFrame(rows).to_csv(str(ROOT / 'outputs' / 'tables' / 'classical_benchmark_results.csv'),index=False)

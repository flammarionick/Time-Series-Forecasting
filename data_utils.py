import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from math import atan2, sqrt
from pathlib import Path

def load_data(path):
    df = pd.read_csv(path)
    # expected columns: ['year','month','day','hour','PM2.5','TEMP','PRES','DEWP','HUMI','Iws','Is','Ir', 'u','v', ...]
    # Parse datetime if not present
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(dict(year=df['year'], month=df['month'], day=df['day'], hour=df['hour']))
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def engineer_features(df):
    d = df.copy()
    # Wind feats
    if {'u','v'}.issubset(d.columns):
        d['wind_speed'] = np.sqrt(d['u']**2 + d['v']**2)
        d['wind_dir']   = np.arctan2(d['v'], d['u'])
        d['wind_sin']   = np.sin(d['wind_dir'])
        d['wind_cos']   = np.cos(d['wind_dir'])
    # Seasonal encodings
    d['hour']      = d['datetime'].dt.hour
    d['dayofweek'] = d['datetime'].dt.dayofweek
    d['month']     = d['datetime'].dt.month
    for k, m in [('hour',24), ('dayofweek',7), ('month',12)]:
        d[f'{k}_sin'] = np.sin(2*np.pi*d[k]/m)
        d[f'{k}_cos'] = np.cos(2*np.pi*d[k]/m)
    # Lags & rolling (shift to avoid leakage)
    for lag in [1,3,6,12,24]:
        d[f'pm_lag{lag}'] = d['PM2.5'].shift(lag)
    d['pm_roll6']  = d['PM2.5'].shift(1).rolling(6).mean()
    d['pm_roll24'] = d['PM2.5'].shift(1).rolling(24).mean()
    d['pm_std24']  = d['PM2.5'].shift(1).rolling(24).std()
    # Imputation for features (target stays NaN where missing)
    feat_cols = [c for c in d.columns if c not in ['datetime','PM2.5']]
    d[feat_cols] = d[feat_cols].interpolate(limit_direction='both')
    return d

def train_val_split(df, val_frac=0.15):
    n = len(df)
    split = int(n*(1-val_frac))
    return df.iloc[:split].copy(), df.iloc[split:].copy()

def make_windows(df, feat_cols, target_col='PM2.5', window=48):
    # drop rows that still have NaN target due to lags
    d  = df.dropna(subset=[target_col]).reset_index(drop=True)
    Xf = d[feat_cols].values; y = d[target_col].values
    X, Y = [], []
    for i in range(window, len(d)):
        X.append(Xf[i-window:i, :]); Y.append(y[i])
    return np.array(X), np.array(Y)

def scale_by_train(X_train, X_val):
    # scale *per feature* across time; reshape to 2D for scaler
    nT, W, F = X_train.shape
    sc = StandardScaler()
    Xtr2 = sc.fit_transform(X_train.reshape(nT*W, F)).reshape(nT, W, F)
    nV, Wv, Fv = X_val.shape
    Xva2 = sc.transform(X_val.reshape(nV*Wv, Fv)).reshape(nV, Wv, Fv)
    return Xtr2, Xva2, sc

import argparse, pandas as pd, numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_utils import load_data, engineer_features, train_val_split, make_windows, scale_by_train
from models import build_lstm
from sklearn.metrics import mean_squared_error
from math import sqrt
from pathlib import Path

def rmse(y, yhat): return sqrt(mean_squared_error(y, yhat))

def main(args):
    df = load_data(args.data)
    df = engineer_features(df)
    # choose features
    feat_cols = [c for c in df.columns if c not in ['datetime','PM2.5','year','day']]
    tr, va = train_val_split(df, val_frac=0.15)

    Xtr, Ytr = make_windows(tr, feat_cols, window=args.window)
    Xva, Yva = make_windows(va, feat_cols, window=args.window)

    Xtr, Xva, scaler = scale_by_train(Xtr, Xva)
    model = build_lstm(Xtr.shape[1:], args.units1, args.units2, args.dropout, args.lr, args.optimizer, args.clipnorm)

    ckpt = ModelCheckpoint(args.model_out, monitor='val_loss', save_best_only=True, verbose=0)
    es   = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

    hist = model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                     epochs=200, batch_size=args.batch, callbacks=[ckpt, es], verbose=0)

    pred = model.predict(Xva, verbose=0).reshape(-1)
    score = rmse(Yva, pred)
    print(f'Val RMSE: {score:.3f}')

    
    Path('runs').mkdir(exist_ok=True)
    row = { 'window': args.window, 'units1': args.units1, 'units2': args.units2, 'dropout': args.dropout,
            'optimizer': args.optimizer, 'lr': args.lr, 'batch': args.batch,
            'clipnorm': args.clipnorm, 'val_rmse': score }
    pd.DataFrame([row]).to_csv('runs/last_run.csv', index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--window', type=int, default=48)
    ap.add_argument('--units1', type=int, default=128)
    ap.add_argument('--units2', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--optimizer', type=str, default='adam')  # adam | adamw | rmsprop
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--clipnorm', type=float, default=1.0)
    ap.add_argument('--model_out', type=str, default='runs/best_lstm.h5')
    args = ap.parse_args()
    main(args)
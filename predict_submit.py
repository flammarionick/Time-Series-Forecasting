import argparse, pandas as pd, numpy as np, tensorflow as tf
from data_utils import load_data, engineer_features, make_windows, scale_by_train
from models import build_lstm

def main(args):
    tr = load_data(args.train); tr = engineer_features(tr)
    te = load_data(args.test);  te = engineer_features(te)

    feat_cols = [c for c in tr.columns if c not in ['datetime','PM2.5','year','day']]
    
    window = args.window
    Xtr, Ytr = make_windows(tr, feat_cols, window=window)
    
    Xtr2, _, scaler = scale_by_train(Xtr, Xtr[:1])  
    # Train final model
    model = build_lstm(Xtr2.shape[1:], args.units1, args.units2, args.dropout, args.lr, args.optimizer, args.clipnorm)
    model.fit(Xtr2, Ytr, epochs= args.epochs, batch_size=args.batch, verbose=0)

   
    cat = pd.concat([tr, te], ignore_index=True)
    Xcat, _ = make_windows(cat, feat_cols, window=window)
    
    n_te = len(te.dropna(subset=['PM2.5']))  
    Xcat2 = scaler.transform(Xcat.reshape(Xcat.shape[0]*Xcat.shape[1], Xcat.shape[2])).reshape(Xcat.shape)

    preds_all = model.predict(Xcat2, verbose=0).reshape(-1)
    
    preds = preds_all[-len(te):]

    sub = pd.read_csv(args.sample)
    sub['PM2.5'] = preds[:len(sub)]
    sub.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--sample', required=True)
    ap.add_argument('--out', default='submissions/sub.csv')
    ap.add_argument('--window', type=int, default=48)
    ap.add_argument('--units1', type=int, default=128)
    ap.add_argument('--units2', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--optimizer', type=str, default='adam')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--clipnorm', type=float, default=1.0)
    args = ap.parse_args()
    main(args)

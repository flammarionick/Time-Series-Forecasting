import itertools, subprocess, pandas as pd, os

grid = {
    'window':   [36, 48, 72],
    'units1':   [96, 128, 256],
    'units2':   [48, 64],
    'dropout':  [0.2, 0.3],
    'optimizer':['adam','adamw'],
    'lr':       [1e-3, 3e-4],
    'batch':    [64, 128]
}
def run(cfg):
    cmd = [
        'python','train_lstm.py',
        '--data','data/train.csv',
        '--window', str(cfg['window']),
        '--units1', str(cfg['units1']),
        '--units2', str(cfg['units2']),
        '--dropout', str(cfg['dropout']),
        '--optimizer', cfg['optimizer'],
        '--lr', str(cfg['lr']),
        '--batch', str(cfg['batch'])
    ]
    subprocess.run(cmd, check=True)
    df = pd.read_csv('runs/last_run.csv')
    for k,v in cfg.items(): df[k] = v
    return df

combos = list(itertools.islice(itertools.product(*grid.values()), 15))  # first 15 combos
keys = list(grid.keys())
rows = []
for tup in combos:
    cfg = dict(zip(keys, tup))
    rows.append(run(cfg))
pd.concat(rows, ignore_index=True).to_csv('experiments.csv', index=False)
print('Wrote experiments.csv')

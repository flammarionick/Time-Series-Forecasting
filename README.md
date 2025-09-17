Setup: pip install -r requirements.txt

Data: place train.csv, test.csv, sample_submission.csv under ./data/

Train: python train_lstm.py --data data/train.csv

Sweep: python sweep.py creates experiments.csv
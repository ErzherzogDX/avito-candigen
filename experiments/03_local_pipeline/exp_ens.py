# Шум ранкера по сидам (±0.2 п.п.) и ансамбль сидов (~0.945).
# Запуск из корня репозитория: python experiments/03_local_pipeline/exp_ens.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks
NQ = int(sys.argv[1])
val = pd.read_parquet('data/val_q.parquet'); nrel = val.rel.str.len().values
def load(name):
    F = pd.read_parquet(f'data/feats_v2_{name}.parquet')
    if name == 'rk': F = F[F.qi < NQ]
    for c in F.columns:
        if F[c].dtype == np.float64: F[c] = F[c].astype(np.float32)
    return add_ranks(F)
tr = load('rk'); va = load('val')
tr = tr[tr.groupby('qi').y.transform('max') > 0].sort_values('qi')
feats = [c for c in tr.columns if c not in ('qi', 'ci', 'y', 'i_cat114')]
def recall(scores):
    d = va[['qi', 'y']].copy(); d['s'] = scores; d['r'] = d.groupby('qi').s.rank(ascending=False, method='first')
    hit = d[(d.r <= 50) & (d.y == 1)].groupby('qi').size()
    return (hit.reindex(range(len(nrel))).fillna(0).values / nrel).mean()
P = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambdarank_truncation_level=60, verbose=-1, num_threads=6)
grp = tr.groupby('qi').size().values; ds = lgb.Dataset(tr[feats], tr.y, group=grp, free_raw_data=False)
preds = []
for sd in range(4):
    m = lgb.train(dict(P, seed=sd), ds, 400); p = m.predict(va[feats]); preds.append(p)
    print('seed', sd, round(recall(p), 4), 'ens so far', round(recall(np.mean(preds, 0)), 4), flush=True)
# rank-average vs score-average

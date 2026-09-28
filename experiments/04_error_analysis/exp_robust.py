# Ранкер без сырых счётчиков: R@35 -0.5 п.п. — счётчики оставлены.
# Запуск из корня репозитория: python experiments/04_error_analysis/exp_robust.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks
NQ = 8000
val = pd.read_parquet('data/val_q.parquet'); nrel = val.rel.str.len().values
def load(name):
    F = pd.read_parquet(f'data/feats_v2_{name}.parquet')
    if name == 'rk': F = F[F.qi < NQ]
    return add_ranks(F)
tr = load('rk'); va = load('val'); tr = tr[tr.groupby('qi').y.transform('max') > 0].sort_values('qi')
def recall(scores, k=50):
    d = va[['qi', 'y']].copy(); d['s'] = scores; d['r'] = d.groupby('qi').s.rank(ascending=False, method='first')
    hit = d[(d.r <= k) & (d.y == 1)].groupby('qi').size()
    return (hit.reindex(range(len(nrel))).fillna(0).values / nrel).mean()
P = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambdarank_truncation_level=60, verbose=-1, num_threads=8)
allf = [c for c in tr.columns if c not in ('qi', 'ci', 'y', 'i_cat114')]
grp = tr.groupby('qi').size().values
for name, drop in [('base', []), ('-raw counts', ['q_Nloc', 'loc_cnt', 'q_nexact', 'hist_n']), ('-raw counts -q_bmax', ['q_Nloc', 'loc_cnt', 'q_nexact', 'hist_n', 'q_bmax'])]:
    fs = [c for c in allf if c not in drop]; r50, r35 = [], []
    for sd in (0, 1, 2):
        p = lgb.train(dict(P, seed=sd), lgb.Dataset(tr[fs], tr.y, group=grp), 400).predict(va[fs])
        r50.append(recall(p)); r35.append(recall(p, 35))
    print(f'{name:22} R@50 {np.mean(r50):.4f}  R@35 {np.mean(r35):.4f}', flush=True)

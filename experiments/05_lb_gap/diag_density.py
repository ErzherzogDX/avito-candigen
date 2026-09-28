# Диагностика разрыва val->LB: зависимость полноты от плотности корпуса (0.964 @0.5x, 0.951 @0.85x, 0.944 @1x).
# Запуск из корня репозитория: python experiments/05_lb_gap/diag_density.py (нужны data/ из scripts/prep.py … split.py)
"""How val recall depends on corpus density: drop random non-relevant items, re-rank, measure; compare with R@k curve."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks
val = pd.read_parquet('data/val_q.parquet'); nrel = val.rel.str.len().values
va0 = pd.read_parquet('data/feats_v2_val.parquet')
m = lgb.Booster(model_file='models/lgb_v2_lambdarank.txt'); fn = m.feature_name()
ncorp = len(np.load('data/val_corpus.npy'))
def rec_at(d, k):
    d = d.copy(); d['s'] = m.predict(d[fn]); d['r'] = d.groupby('qi').s.rank(ascending=False, method='first')
    hit = d[(d.r <= k) & (d.y == 1)].groupby('qi').size()
    return (hit.reindex(range(len(nrel))).fillna(0).values / nrel).mean()
full = add_ranks(va0.copy())
print('R@k at full density:', {k: round(rec_at(full, k), 4) for k in [25, 30, 35, 40, 50]})
rng = np.random.default_rng(0)
for keep in [0.5, 0.7, 0.85]:
    drop = rng.random(ncorp) > keep
    d = va0[(va0.y == 1) | ~drop[va0.ci.values]].copy()
    d = add_ranks(d)
    print(f'density x{keep:.2f}: R@50 = {rec_at(d, 50):.4f}')

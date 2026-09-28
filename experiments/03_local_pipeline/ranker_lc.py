# Кривая обучения ранкера и сравнение objective: lambdarank 0.944, xendcg 0.916, binary 0.895.
# Запуск из корня репозитория: python experiments/03_local_pipeline/ranker_lc.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks
tr = add_ranks(pd.read_parquet('data/feats_v2_rk.parquet')); va = add_ranks(pd.read_parquet('data/feats_v2_val.parquet'))
val = pd.read_parquet('data/val_q.parquet'); nrel = val.rel.str.len().values
feats = [c for c in tr.columns if c not in ('qi', 'ci', 'y', 'i_cat114')]
tr = tr[tr.groupby('qi').y.transform('max') > 0].sort_values('qi')
def recall(scores):
    d = va[['qi', 'y']].copy(); d['s'] = scores; d['r'] = d.groupby('qi').s.rank(ascending=False, method='first')
    hit = d[(d.r <= 50) & (d.y == 1)].groupby('qi').size()
    return (hit.reindex(range(len(nrel))).fillna(0).values / nrel).mean()
base = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8)
def fit(sub, obj, n=300, **kw):
    p = dict(base, objective=obj, **kw)
    ds = lgb.Dataset(sub[feats], sub.y, group=sub.groupby('qi').size().values if obj in ('lambdarank', 'rank_xendcg') else None)
    return lgb.train(p, ds, num_boost_round=n)
qs = tr.qi.unique()
for nq in [4000, 8000, len(qs)]:
    sub = tr[tr.qi.isin(qs[:nq])]
    print('nq', nq, 'lambdarank', round(recall(fit(sub, 'lambdarank', lambdarank_truncation_level=60).predict(va[feats])), 4), flush=True)
print('binary', round(recall(fit(tr, 'binary').predict(va[feats])), 4), flush=True)
print('xendcg', round(recall(fit(tr, 'rank_xendcg').predict(va[feats])), 4), flush=True)
print('lambdarank lr.03 n600 leaves127', round(recall(fit(tr, 'lambdarank', n=600, lambdarank_truncation_level=60, learning_rate=0.03, num_leaves=127).predict(va[feats])), 4), flush=True)
print('lambdarank trunc 100', round(recall(fit(tr, 'lambdarank', lambdarank_truncation_level=100).predict(va[feats])), 4), flush=True)

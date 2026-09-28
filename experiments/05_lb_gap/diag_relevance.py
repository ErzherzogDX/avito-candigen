# Диагностика разрыва val->LB: определение релевантности (text+loc = 0.944, только текст = 0.795) и перевзвешивание под состав бенчмарка (0.941).
# Запуск из корня репозитория: python experiments/05_lb_gap/diag_relevance.py (нужны data/ из scripts/prep.py … split.py)
"""Val recall under alternative relevance definitions + bench-like reweighting."""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks
# reconstruct V log exactly as split.py
rng = np.random.default_rng(42)
items = pd.read_parquet('data/items.parquet', columns=['iid', 'item_id', 'item_location_id'])
log = pd.read_parquet('data/log.parquet')
h = items.item_id.str[-6:].map(lambda s: int(s, 16)) % 1000
isV = (h < 640).values; shared = isV & (rng.random(len(items)) < 0.45); coin = rng.random(len(log)) < 0.5
vi, sh = isV[log.iid.values], shared[log.iid.values]
V = log[vi & ~(sh & coin)]
val = pd.read_parquet('data/val_q.parquet'); corpus = np.load('data/val_corpus.npy')
va = add_ranks(pd.read_parquet('data/feats_v2_val.parquet'))
m = lgb.Booster(model_file='models/lgb_v2_lambdarank.txt'); va['s'] = m.predict(va[m.feature_name()])
va['r'] = va.groupby('qi').s.rank(ascending=False, method='first')
top = va[va.r <= 50].groupby('qi').ci.apply(lambda c: set(corpus[c.values]))
top = top.reindex(range(len(val))).apply(lambda x: x if isinstance(x, set) else set())
rel_text = V.groupby('search_query').iid.apply(lambda s: set(s))
rel_tl = V.groupby(['search_query', 'search_location_id']).iid.apply(lambda s: set(s))
def rec(rels): return np.array([len(r & t) / len(r) for r, t in zip(rels, top)])
r_key = rec([set(r) for r in val.rel])
r_text = rec([rel_text[q] for q in val.search_query])
r_tl = rec([rel_tl[(q, l)] for q, l in zip(val.search_query, val.search_location_id)])
print('val R@50  full-key rel:', r_key.mean().round(4), '| text+loc rel:', r_tl.mean().round(4), '| text-only rel:', r_text.mean().round(4))
print('mean #rel  key', val.rel.str.len().mean().round(3), 'text+loc', np.mean([len(rel_tl[(q, l)]) for q, l in zip(val.search_query, val.search_location_id)]).round(3),
      'text', np.mean([len(rel_text[q]) for q in val.search_query]).round(3))
# bench-like reweighting: location mix + empty-filter share
bq = pd.read_parquet('data/bench_q.parquet')
def seg(df):
    big = df.search_location_id.map({637640: 'MSK', 653240: 'SPB', 107620: 'MSKreg', 107621: 'SPBreg', 621540: 'RU'}).fillna('other')
    return big + '|' + np.where(df.search_infm_params_text == '', 'nof', 'f')
sv, sb = seg(val), seg(bq)
w = (sb.value_counts(normalize=True) / sv.value_counts(normalize=True)).reindex(sv).fillna(0).values
print('bench-reweighted val R@50 (full-key rel):', np.average(r_key, weights=w).round(4))
print(pd.DataFrame({'val_share': sv.value_counts(normalize=True), 'bench_share': sb.value_counts(normalize=True),
                    'val_R': pd.Series(r_key).groupby(sv.values).mean()}).round(3).sort_values('bench_share', ascending=False))

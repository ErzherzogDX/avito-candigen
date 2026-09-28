# Инференс локальной версии на бенчмарке + правила категорий (попытка 1, LB 0.9163).
# Запуск из корня репозитория: python experiments/03_local_pipeline/predict_bench.py (нужны data/ из scripts/prep.py … split.py)
"""Bench inference: corpus = benchmark_items, log = full train.
usage: predict_bench.py OUT_CSV LGB_MODEL dense_names(comma) [feats_tag]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import Builder, add_ranks
out, model_path = sys.argv[1], sys.argv[2]
dense_names = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] else []
ftag = sys.argv[4] if len(sys.argv) > 4 else 'bench'
knn_tag = sys.argv[5] if len(sys.argv) > 5 else None
R0 = int(sys.argv[6]) if len(sys.argv) > 6 else 15      # max reserved slots for non-114 items on category-0 queries
items = pd.read_parquet('data/items.parquet', columns=['iid', 'item_id', 'in_corpus'])
corpus = np.sort(items.iid.values[items.in_corpus.values == 1])
bq = pd.read_parquet('data/bench_q.parquet')
import os
fp = f'data/feats_{ftag}.parquet'
if os.path.exists(fp):
    F = pd.read_parquet(fp)
else:
    log = pd.read_parquet('data/log.parquet')
    knn = (np.load('data/logQF_texts.npy', allow_pickle=True), np.load(f'data/emb_{knn_tag}_logQF.npy')) if knn_tag else None
    B = Builder(corpus, log, knn=knn)
    # dense spec 'feat=files' lets a ranker trained on e5sT features consume e5sF embeddings
    dense = {}
    for spec in dense_names:
        fn_, src_ = spec.split('=') if '=' in spec else (spec, spec)
        dense[fn_] = (np.load(f'data/emb_{src_}_benchQ.npy').astype(np.float32), np.load(f'data/emb_{src_}_benchcorpus.npy').astype(np.float32))
    qplain = np.load(f'data/emb_{knn_tag}_benchQp.npy').astype(np.float32) if knn_tag else None
    F = B.build(bq, dense, with_labels=False, qplain=qplain)
    F.to_parquet(fp, index=False)
F = add_ranks(F)
ms = [lgb.Booster(model_file=p) for p in model_path.split(',')]
F['s'] = np.mean([m.predict(F[m.feature_name()]) for m in ms], axis=0)
# category rules: cat-114 search -> only cat-114 items; cat-0 search -> reserve slots for text-matching non-114 items
qcat = bq.search_category.values[F.qi.values]
F = F[~((qcat == 114) & (F.i_cat114.values == 0))]
F = F.sort_values(['qi', 's'], ascending=[True, False])
dn_rk = [c for c in F.columns if c.startswith('rk_dn_')]
qcat = bq.search_category.values[F.qi.values]
non = F[(qcat == 0) & (F.i_cat114.values == 0)]
good = (non.bmn_all >= 0.4) | (non.rk_bm_all <= 20)
for c in dn_rk: good |= non[c] <= 20
res_non = non[good].groupby('qi').head(R0)
rest = F.drop(res_non.index)
k_rest = (50 - res_non.groupby('qi').size()).reindex(range(len(bq))).fillna(50).astype(int)
rest = rest[rest.groupby('qi').cumcount() < k_rest.values[rest.qi.values]]
top = pd.concat([res_non, rest]).sort_values(['qi', 's'], ascending=[True, False])
print('cat0 queries with reserved non-114:', res_non.qi.nunique(), 'slots', len(res_non))
ids = items.item_id.values
ans = top.groupby('qi').ci.apply(lambda c: ' '.join(ids[corpus[c.values]]))
res = pd.DataFrame({'query_id': bq.query_id.values, 'answer': ans.reindex(range(len(bq))).fillna('').values})
res.to_csv(out, index=False)
print('written', out, res.shape)

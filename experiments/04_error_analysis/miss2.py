# Разбор промахов v2: 81% не попавших в кандидаты — другие локации; однословные запросы хуже всех (0.916).
# Запуск из корня репозитория: python experiments/04_error_analysis/miss2.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'src'); from pipeline import add_ranks
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
va = add_ranks(pd.read_parquet('data/feats_v2_val.parquet')); val = pd.read_parquet('data/val_q.parquet')
m = lgb.Booster(model_file='models/lgb_v2_lambdarank.txt'); va['s'] = m.predict(va[m.feature_name()])
va['r'] = va.groupby('qi').s.rank(ascending=False, method='first')
corpus = np.load('data/val_corpus.npy'); items = pd.read_parquet('data/items.parquet', columns=['iid','item_title_raw','item_location_id','item_microcat_id'])
val['nrel'] = val.rel.str.len()
pos = va[va.y==1]
val['hit'] = pos[pos.r<=50].groupby('qi').size().reindex(range(len(val))).fillna(0).values
val['inc'] = pos.groupby('qi').size().reindex(range(len(val))).fillna(0).values
val['rec'] = val.hit/val.nrel; val['hasf'] = val.search_infm_params_text!=''
print(val.groupby(['seen']).rec.mean().round(4).to_dict(), val.groupby(['hasf']).rec.mean().round(4).to_dict())
val['ntok'] = val.search_query.str.split().str.len().clip(upper=5)
print(val.groupby('ntok').rec.agg(['mean','size']).round(3))
# rank distribution of positives among misses
print('pos rank buckets', pd.cut(pos.r, [0,50,60,80,100,200,1000]).value_counts().sort_index().to_dict())
# for misses in candidates: which features differ
mr = pos[pos.r>50]
print('missed-in-cand: same_loc', mr.same_loc.mean().round(3), 'vs hits', pos[pos.r<=50].same_loc.mean().round(3))
print('missed-in-cand: dist med', mr.dist.median().round(3), 'vs hits', pos[pos.r<=50].dist.median().round(3))
print('missed-in-cand: rk_dn med', mr.rk_dn_e5sT.median(), 'rk_bm_all med', mr.rk_bm_all.median())
# not-in-cand: where are they?
nc = val[val.inc < val.nrel]
iloc = items.item_location_id.values
nc_same = np.mean([iloc[r[0]]==s for r, s in zip(nc.rel, nc.search_location_id)])
print('not-in-cand queries', len(nc), 'same-loc share', round(nc_same,3))

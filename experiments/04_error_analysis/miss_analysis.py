# Разбор промахов: семантика без лексического совпадения, опечатки, соседние локации.
# Запуск из корня репозитория: python experiments/04_error_analysis/miss_analysis.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd, lightgbm as lgb
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
va = pd.read_parquet('data/feats_zs_val.parquet'); val = pd.read_parquet('data/val_q.parquet')
m = lgb.Booster(model_file='models/lgb_zs_lambdarank.txt'); feats = m.feature_name()
va['s'] = m.predict(va[feats]); va['r'] = va.groupby('qi').s.rank(ascending=False, method='first')
corpus = np.load('data/val_corpus.npy'); items = pd.read_parquet('data/items.parquet', columns=['iid','item_title_raw','item_location_id','item_microcat_id'])
val['nrel'] = val.rel.str.len()
hit = va[(va.r<=50)&(va.y==1)].groupby('qi').size(); incand = va[va.y==1].groupby('qi').size()
val['hit'] = hit.reindex(range(len(val))).fillna(0).values; val['incand'] = incand.reindex(range(len(val))).fillna(0).values
val['rec'] = val.hit/val.nrel
val['hasf'] = val.search_infm_params_text!=''
print(val.groupby('seen').rec.mean(), val.groupby('hasf').rec.mean())
w = val.groupby(['seen','hasf']).rec.mean(); print(w)
# bench-weighted: seen 37%, hasf 37%
print('overall', val.rec.mean())
miss_nc = val[val.incand < val.nrel]
print('queries w/ rel not in cands', len(miss_nc))
ex = miss_nc.sample(25, random_state=0)
for r in ex.itertuples():
    it = items.set_index('iid').loc[r.rel]
    print(f'{r.search_query!r:45} sloc={r.search_location_id} f={r.search_infm_params_text[:30]!r} -> ', list(zip(it.item_title_raw.str[:50], it.item_location_id)))
print('-----ranked out')
miss_rk = val[(val.incand == val.nrel) & (val.hit < val.nrel)]
print('queries w/ rel in cands but rank>50', len(miss_rk))
for r in miss_rk.sample(20, random_state=0).itertuples():
    ci = np.searchsorted(corpus, r.rel)
    rr = va[(va.qi==r.Index)&(va.ci.isin(ci))].r.values
    top = va[(va.qi==r.Index)&(va.r<=3)].ci.values
    print(f'{r.search_query!r:40} rank={rr} rel={items.item_title_raw.values[r.rel[0]][:45]!r} top={[items.item_title_raw.values[corpus[c]][:30] for c in top]}')

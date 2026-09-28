# Потолок полноты при отборе по локации: та же локация 77%, локации с P>=1% — 92% при медианном пуле ~3k.
# Запуск из корня репозитория: python experiments/01_eda/loc_study.py (нужны data/ из scripts/prep.py … split.py)
import pandas as pd, numpy as np
items = pd.read_parquet('data/items.parquet', columns=['iid','item_location_id','item_latitude','item_longitude','item_category_id'])
T = pd.read_parquet('data/T_log.parquet'); val = pd.read_parquet('data/val_q.parquet'); corpus = np.load('data/val_corpus.npy')
iloc = items.item_location_id.values
T['il'] = iloc[T.iid.values]
pair = T.groupby(['search_location_id','il']).size().rename('n').reset_index()
tot = pair.groupby('search_location_id').n.transform('sum'); pair['p']=pair.n/tot
cl = pd.Series(iloc[corpus]).value_counts()
val_rel = val.explode('rel'); val_rel['rloc'] = iloc[val_rel.rel.astype(int).values]
print('same loc recall', (val_rel.rloc==val_rel.search_location_id).mean())
pm = {k: dict(zip(g.il, g.p)) for k,g in pair.groupby('search_location_id')}
for th in [0.2,0.05,0.02,0.01,0.005,0.002,0.0]:
    hit=[]; sizes=[]
    for q in val.itertuples():
        d = pm.get(q.search_location_id, {})
        locs = {l for l,p in d.items() if p>=th} | {q.search_location_id}
        sizes.append(sum(cl.get(l,0) for l in locs))
    val['pool']=sizes
    vr = val_rel.merge(val[['qid']], on='qid')
    ok = [ (r.rloc in ({l for l,p in pm.get(r.search_location_id,{}).items() if p>=th}|{r.search_location_id})) for r in val_rel.itertuples()]
    print(f'th={th}: loc recall {np.mean(ok):.4f}  pool median {np.median(sizes):.0f} p90 {np.percentile(sizes,90):.0f} max {max(sizes)}')
print('val search locs unseen in T', (~val.search_location_id.isin(pm.keys())).mean())

# EDA-6: корпус бенчмарка ~ «выбранные объявления отложенной части лога» (распределение локаций как у train x0.55).
# Запуск из корня репозитория: python experiments/01_eda/eda6.py (нужны data/ из scripts/prep.py … split.py)
import pandas as pd, numpy as np
tr = pd.read_parquet('train.parquet', columns=['search_query','search_location_id','item_id','item_location_id'])
bq = pd.read_parquet('benchmark_queries.parquet')
bi = pd.read_parquet('benchmark_items.parquet', columns=['item_id','item_location_id','item_category_id'])
# location mapping from train: P(item_loc | search_loc)
pair = tr.groupby(['search_location_id','item_location_id']).size().rename('n').reset_index()
pair['p'] = pair.n / pair.groupby('search_location_id').n.transform('sum')
# for each bench query, set of plausible item locs (p>=0.01)
bl = set(bq.search_location_id)
plaus = pair[(pair.search_location_id.isin(bl)) & (pair.p>=0.005)]
plaus_locs = set(plaus.item_location_id)
print('n plausible item locs for bench', len(plaus_locs))
print('corpus share in plausible locs', bi.item_location_id.isin(plaus_locs).mean())
tri = tr.drop_duplicates('item_id')
print('train items share in plausible locs', tri.item_location_id.isin(plaus_locs).mean())
c0 = bq[bq.search_category==0]
pl0 = set(pair[(pair.search_location_id.isin(set(c0.search_location_id))) & (pair.p>=0.005)].item_location_id)
n = bi[bi.item_category_id!=114]
print('non114 corpus share in cat0-plausible locs', n.item_location_id.isin(pl0).mean(), ' vs 114 corpus share', bi[bi.item_category_id==114].item_location_id.isin(pl0).mean())
# per-location ratio corpus/train items
cl = bi.item_location_id.value_counts(); tl = tri.item_location_id.value_counts()
r = pd.concat([cl.rename('corpus'), tl.rename('train')], axis=1).fillna(0)
r['bench_q'] = bq.search_location_id.value_counts().reindex(r.index).fillna(0)
r['ratio'] = r.corpus/(r.train+1)
print(r.sort_values('corpus', ascending=False).head(25))
print('corr ratio with bench_q>0:', r.groupby(r.bench_q>0).ratio.describe())

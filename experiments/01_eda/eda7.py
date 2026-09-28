# EDA-7: бенчмарк ~ равномерная выборка по уникальным текстам; доля пустых фильтров по частоте запроса.
# Запуск из корня репозитория: python experiments/01_eda/eda7.py (нужны data/ из scripts/prep.py … split.py)
import pandas as pd, numpy as np
tr = pd.read_parquet('train.parquet', columns=['search_query','search_location_id','search_infm_params_text','search_category','item_id'])
bq = pd.read_parquet('benchmark_queries.parquet')
tr['emp'] = tr.search_infm_params_text.fillna('')==''
f = tr.search_query.map(tr.search_query.value_counts())
for lo,hi in [(1,1),(2,3),(4,10),(11,100),(101,10**6)]:
    m=(f>=lo)&(f<=hi); print(lo,hi,'rows',m.sum(),'empty share',tr.emp[m].mean().round(3))
# one random row per unique text
one = tr.sample(frac=1, random_state=0).drop_duplicates('search_query')
print('uniform-over-texts empty share', one.emp.mean())
print('uniform-over-texts moscow share', (one.search_location_id==637640).mean(), 'bench', (bq.search_location_id==637640).mean())
bq['len']=bq.search_query.str.split().str.len(); one['len']=one.search_query.str.split().str.len()
print('query word len bench', bq.len.describe().round(2).to_dict()); print('train uniform', one.len.describe().round(2).to_dict())
tf = tr.search_query.value_counts()
bq['trfreq']=bq.search_query.map(tf).fillna(0)
print('bench text freq in train', bq.trfreq.describe(percentiles=[.5,.63,.7,.8,.9,.95]).round(1).to_dict())
# items per full key
key=['search_query','search_location_id','search_infm_params_text','search_category']
g=tr.groupby(key,dropna=False).item_id.nunique()
print('uniq items per key', g.value_counts().sort_index().head(10).to_dict())
g1 = one.set_index(key).index.map(g)
print('uniq items per key for uniform-by-text sample', pd.Series(g1).value_counts().sort_index().head(10).to_dict(), pd.Series(g1).mean())

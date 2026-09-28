# EDA-2: структура локаций (регион -> город), плотность корпуса по локациям, запросы с search_category=0.
# Запуск из корня репозитория: python experiments/01_eda/eda2.py (нужны data/ из scripts/prep.py … split.py)
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 100)
tr = pd.read_parquet('train.parquet', columns=['search_query','search_location_id','search_category','item_id','item_location_id','item_title_raw','item_microcat_id'])
bq = pd.read_parquet('benchmark_queries.parquet')
bi = pd.read_parquet('benchmark_items.parquet', columns=['item_id','item_title_raw','item_location_id','item_microcat_id','item_category_id','item_latitude','item_longitude'])
locc = bi.item_location_id.value_counts()
print('corpus items per location: top\n', locc.head(15))
bq['n_same_loc'] = bq.search_location_id.map(locc).fillna(0)
print('corpus items in same loc as bench query:', bq.n_same_loc.describe(percentiles=[.1,.25,.5,.75,.9]))
print('bench queries loc vc\n', bq.search_location_id.value_counts().head(15))
# train: same loc distribution for items per query location
trl = tr.search_location_id.value_counts()
print('train loc vc\n', trl.head(10))
# is the corpus concentrated on bench locations?
print('share corpus items in bench query locations', bi.item_location_id.isin(set(bq.search_location_id)).mean())
# train location mismatch: what does mismatch look like
mm = tr[tr.search_location_id!=tr.item_location_id]
print(mm[['search_query','search_location_id','item_location_id','item_title_raw']].head(20))
print('mismatch pairs top\n', mm.groupby(['search_location_id','item_location_id']).size().sort_values(ascending=False).head(20))
# rows per unique query text in train
print('train query text freq', tr.search_query.value_counts().describe(percentiles=[.5,.9,.99]))
print(tr.search_query.value_counts().head(20))
# category 0 in bench
print(bq[bq.search_category==0].head(20))

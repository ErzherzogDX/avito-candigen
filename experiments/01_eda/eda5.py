# EDA-5: объявления вне категории 114 и запросы с search_category=0.
# Запуск из корня репозитория: python experiments/01_eda/eda5.py (нужны data/ из scripts/prep.py … split.py)
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70); pd.set_option('display.max_rows', 200)
tr = pd.read_parquet('train.parquet', columns=['search_query','search_location_id','search_category','item_category_id','item_title_raw','item_location_id'])
bq = pd.read_parquet('benchmark_queries.parquet')
bi = pd.read_parquet('benchmark_items.parquet', columns=['item_id','item_title_raw','item_location_id','item_category_id','item_microcat_id'])
print(tr[tr.search_category!=114])
print(tr[tr.item_category_id!=114])
n = bi[bi.item_category_id!=114]
print(n.sample(40, random_state=0)[['item_title_raw','item_category_id','item_location_id']])
c0 = bq[bq.search_category==0]
print(c0.sample(60, random_state=1)[['search_query','search_location_id','search_infm_params_text']])
print('cat0 queries with infm nonempty', (c0.search_infm_params_text.fillna('')!='').mean())

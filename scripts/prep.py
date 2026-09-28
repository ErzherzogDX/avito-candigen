"""Build unified item table and query log with integer ids.

Шаг 1 подготовки данных. Объявления из train и из корпуса бенчмарка сводятся в одну таблицу data/items.parquet
с целочисленным идентификатором iid (все дальнейшие матрицы индексируются по нему); лог «запрос -> выбранное объявление»
сохраняется в data/log.parquet, запросы бенчмарка — в data/bench_q.parquet (порядок строк как в исходном файле).
"""
import pandas as pd, numpy as np
ICOLS = ['item_id','item_title_raw','item_description_raw','item_infm_params_text','item_category_id','item_microcat_id',
         'item_price','item_rating','item_rating_reviews_count','item_location_id','item_latitude','item_longitude',
         'item_is_phone_hidden','item_is_message_forbidden']
QCOLS = ['search_query','search_location_id','search_is_delivery_search','search_infm_params_text','search_category']
tr = pd.read_parquet('train.parquet')
bi = pd.read_parquet('benchmark_items.parquet')
ti = tr[ICOLS].drop_duplicates('item_id')
# сначала корпус бенчмарка (in_corpus=1), затем объявления, которые есть только в train
items = pd.concat([bi[ICOLS].assign(in_corpus=1), ti[~ti.item_id.isin(set(bi.item_id))].assign(in_corpus=0)], ignore_index=True)
items['in_train'] = items.item_id.isin(set(tr.item_id)).astype(np.int8)   # ~9.6% корпуса бенчмарка уже встречались в train
for c in ['item_price','item_latitude','item_longitude']: items[c] = items[c].astype(float)   # decimal -> float
for c in ['item_title_raw','item_description_raw','item_infm_params_text']: items[c] = items[c].fillna('')
items['iid'] = np.arange(len(items), dtype=np.int32)
items.to_parquet('data/items.parquet', index=False)
print(items.shape, items.in_corpus.sum(), items.in_train.sum())
log = tr[QCOLS+['item_id']].copy()
log['search_infm_params_text'] = log.search_infm_params_text.fillna('')
log['iid'] = log.item_id.map(pd.Series(items.iid.values, index=items.item_id)).astype(np.int32)
log.drop(columns='item_id').to_parquet('data/log.parquet', index=False)
bq = pd.read_parquet('benchmark_queries.parquet'); bq['search_infm_params_text']=bq.search_infm_params_text.fillna('')
bq.to_parquet('data/bench_q.parquet', index=False)
# sanity: are item features consistent across train rows?
x = tr.groupby('item_id').item_title_raw.nunique(); print('items with >1 title', (x>1).sum())

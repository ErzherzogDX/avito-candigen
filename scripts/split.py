"""Pseudo-benchmark: item-disjoint split of train log into T (training) and V (val corpus + queries).

Шаг 4 подготовки данных — локальная валидация, повторяющая устройство бенчмарка:
  * корпус бенчмарка — это «все выбранные объявления из отложенной части лога»: лишь ~5% объявлений train есть в нём,
    ~9.6% его объявлений встречались в train. Поэтому делим ОБЪЯВЛЕНИЯ train по хешу item_id:
    64% -> V (они образуют val-корпус ~178k, близко к 189k бенчмарка), остальные -> T (лог для обучения всех моделей);
  * часть V-объявлений (45%) «общие»: их строки лога случайно делятся между T и V -> ~8% val-корпуса имеет историю в T (как 9.6% в бенчмарке);
  * запросы бенчмарка уникальны по тексту и в 37% встречаются в train -> берём по одному ключу запроса на текст
    и собираем 3000 val-запросов с той же долей «виденных» текстов; остальные (~45k) — для обучения ранкера (rk).
Разбиение детерминировано (фиксированные сиды), на любой машине получается одинаковым.
"""
import pandas as pd, numpy as np
rng = np.random.default_rng(42)
items = pd.read_parquet('data/items.parquet', columns=['iid','item_id','in_train'])
log = pd.read_parquet('data/log.parquet')
h = items.item_id.str[-6:].map(lambda s: int(s,16)) % 1000        # item_id — hex-строка: детерминированный хеш
isV = (h < 640).values
shared = isV & (rng.random(len(items)) < 0.45)          # ~10% of V items also have history in T
log['v_item'] = isV[log.iid.values]
log['sh'] = shared[log.iid.values]
coin = rng.random(len(log)) < 0.5
log['part'] = np.where(~log.v_item, 'T', np.where(log.sh & coin, 'T', 'V'))
T = log[log.part=='T'].drop(columns=['v_item','sh','part']).reset_index(drop=True)
V = log[log.part=='V'].drop(columns=['v_item','sh','part']).reset_index(drop=True)
corpus = np.unique(V.iid.values)
print('T rows', len(T), 'V rows', len(V), 'val corpus', len(corpus), 'corpus seen in T', np.isin(corpus, T.iid.values).mean())
# «запрос» = полный ключ (текст, локация, доставка, фильтры, категория); релевантные — все V-объявления, выбранные по нему
KEY = ['search_query','search_location_id','search_is_delivery_search','search_infm_params_text','search_category']
g = V.groupby(KEY).iid.apply(lambda s: sorted(set(s))).reset_index().rename(columns={'iid':'rel'})
g = g.sample(frac=1, random_state=1).drop_duplicates('search_query').reset_index(drop=True)   # one key per text
tset = set(T.search_query)
g['seen'] = g.search_query.isin(tset)
print('texts in V', len(g), 'seen share', g.seen.mean())
# val: 3000 queries with bench-like 37% seen; ranker-train: the rest (keep seen/unseen as is, cap)
nv = 3000
val = pd.concat([g[g.seen].sample(int(nv*.37), random_state=2), g[~g.seen].sample(nv-int(nv*.37), random_state=3)])
rest = g.drop(val.index)
val = val.reset_index(drop=True); rest = rest.reset_index(drop=True)
print('val empty filters', (val.search_infm_params_text=='').mean(), 'n rel', val.rel.str.len().mean())
print('rest', len(rest), 'seen', rest.seen.mean())
val['qid'] = ['v%05d'%i for i in range(len(val))]; rest['qid'] = ['r%05d'%i for i in range(len(rest))]
T.to_parquet('data/T_log.parquet', index=False)
val.to_parquet('data/val_q.parquet', index=False); rest.to_parquet('data/rk_q.parquet', index=False)
np.save('data/val_corpus.npy', corpus)

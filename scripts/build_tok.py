"""Tokenize all items (title, clean params, description) once; cache as pickle.

Шаг 2 подготовки данных: основы слов по трём полям для BM25 (заголовок / параметры без адреса, цен и расписания / описание)
и очищенный текст параметров (ptxt) для текста энкодеров. Считается один раз (~1 мин) для всех 516k объявлений.
"""
import sys, pickle, pandas as pd, numpy as np, time
sys.path.insert(0, 'src'); from common import *
t0=time.time()
items = pd.read_parquet('data/items.parquet', columns=['iid','item_title_raw','item_description_raw','item_infm_params_text'])
title = [toks(s) for s in items.item_title_raw]
ptxt = [param_text_clean(s) for s in items.item_infm_params_text]
par = [toks(s) for s in ptxt]
desc = [toks(s) for s in items.item_description_raw]
pickle.dump({'title':title,'par':par,'desc':desc,'ptxt':ptxt}, open('data/item_toks.pkl','wb'))
print('done', time.time()-t0)

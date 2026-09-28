"""Compact item text for dense encoders.

Шаг 3 подготовки данных: «заголовок. параметры услуги. первые 300 символов описания» — вход bi-encoder и cross-encoder
(в 128 токенов помещается главное; длинные описания в основном повторяются).
"""
import sys, pickle, pandas as pd
sys.path.insert(0, 'src'); from common import *
items = pd.read_parquet('data/items.parquet', columns=['iid','item_title_raw','item_description_raw'])
tk = pickle.load(open('data/item_toks.pkl','rb'))
def mk(t, p, d, dl=300):
    d = ' '.join(d.split())[:dl]
    return f'{t}. {p}. {d}'
txt = [mk(t, p, d) for t, p, d in zip(items.item_title_raw, tk['ptxt'], items.item_description_raw)]
pd.DataFrame({'iid': items.iid, 'text': txt}).to_parquet('data/item_text.parquet', index=False)
print(txt[:3])

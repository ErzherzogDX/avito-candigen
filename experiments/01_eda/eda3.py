# EDA-3: тексты фильтров поиска и параметров объявлений; выполняется ли фильтр по рейтингу.
# Запуск из корня репозитория: python experiments/01_eda/eda3.py (нужны data/ из scripts/prep.py … split.py)
import pandas as pd, numpy as np, re
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 200)
tr = pd.read_parquet('train.parquet', columns=['search_query','search_location_id','search_infm_params_text','item_id','item_infm_params_text','item_rating','item_rating_reviews_count','item_microcat_id','item_title_raw','item_description_raw','item_price'])
tr['sp'] = tr.search_infm_params_text.fillna('')
print(tr.sp.value_counts().head(40))
# sample item params
for s in tr.item_infm_params_text.sample(8, random_state=1): print(repr(s[:400]))
print('--- desc lens', tr.item_description_raw.fillna('').str.len().describe())
print('--- title lens', tr.item_title_raw.fillna('').str.len().describe())
# rating filter
m = tr.sp.str.contains('Рейтинг пользователя 4 звезды')
print('rating filter rows', m.sum(), 'item rating>=4 share', (tr.loc[m,'item_rating']>=4).mean(), 'nan share', tr.loc[m,'item_rating'].isna().mean())
print('no filter: rating>=4', (tr.loc[~m,'item_rating']>=4).mean(), 'nan', tr.loc[~m,'item_rating'].isna().mean())
# Вид услуги X filter: check item params contain it
def vid(s):
    mm = re.search(r'Вид услуги ([^А-ЯЁ]*?[А-ЯЁ][^А-ЯЁ]*?)(?= [А-ЯЁ]|$)', s)
    return mm.group(1) if mm else None

# EDA-4: совпадение фильтров «Вид услуги»/«Тип услуги» с параметрами выбранного объявления (98.5% / 94.6%).
# Запуск из корня репозитория: python experiments/01_eda/eda4.py (нужны data/ из scripts/prep.py … split.py)
import pandas as pd, numpy as np, re
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 200)
tr = pd.read_parquet('train.parquet', columns=['search_query','search_infm_params_text','item_infm_params_text','item_microcat_id'])
tr['sp'] = tr.search_infm_params_text.fillna(''); tr['ip']=tr.item_infm_params_text.fillna('')
# item Вид услуги value: text between 'Вид услуги ' and next known key
keys = ['Место оказания услуг','Тип услуги','Тип стоимости','Гарантия','Опыт работы','Где вы оказываете','Название услуги','Специальность','Время для связи','График работы','Онлайн-запись','Ваши клиенты','Кто оказывает','Дополнительно','Начальная цена','Услуга','Стоимость','Вид услуги']
kre = '|'.join(map(re.escape, keys))
def get(s, key):
    m = re.search(re.escape(key)+r' (.*?)(?= (?:'+kre+r')\b|$)', s)
    return m.group(1).strip() if m else None
samp = tr.sample(200000, random_state=0)
samp['i_vid'] = samp.ip.map(lambda s: get(s,'Вид услуги'))
samp['s_vid'] = samp.sp.map(lambda s: get(s,'Вид услуги'))
samp['i_tip'] = samp.ip.map(lambda s: get(s,'Тип услуги'))
samp['s_tip'] = samp.sp.map(lambda s: get(s,'Тип услуги'))
print(samp.i_vid.value_counts().head(40))
x = samp[samp.s_vid.notna() & (samp.s_vid!='')]
print('rows with search vid', len(x), 'match item vid', (x.s_vid==x.i_vid).mean())
print(x[x.s_vid!=x.i_vid][['search_query','s_vid','i_vid']].head(20))
x = samp[samp.s_tip.notna() & (samp.s_tip!='')]
print('rows with search tip', len(x), 'match item tip', (x.s_tip==x.i_tip).mean())
print(x[x.s_tip!=x.i_tip][['search_query','s_tip','i_tip']].head(20))
# microcat vs vid
print(pd.crosstab(samp.i_vid, samp.item_microcat_id).shape)
print(samp.groupby('item_microcat_id').i_vid.agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else np.nan).describe())

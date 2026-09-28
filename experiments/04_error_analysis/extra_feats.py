# Признаки «нелокальности» запроса/микрокатегории и «удалённой» услуги (для exp_extra.py).
# Запуск из корня репозитория: python experiments/04_error_analysis/extra_feats.py (нужны data/ из scripts/prep.py … split.py)
"""Post-hoc features: query/microcat 'far-rate' (how often chosen item is outside search loc), item remote-service flags."""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, 'src'); from common import toks, norm

REMOTE = re.compile(r'онлайн|удален|удалён|дистанц|по всей росси|по всей стране|по россии|любой город|любом городе|доставк|отправк|почтой|сдэк|cdek|zoom|skype|телеграм|whatsapp')


class Extra:
    def __init__(self, corpus, log, items):
        il = items.item_location_id.values; mc = items.item_microcat_id.values
        L = log[['search_query', 'search_location_id', 'iid']].copy()
        L['far'] = (il[L.iid.values] != L.search_location_id.values).astype(np.float32)
        L['mc'] = mc[L.iid.values]
        g = L.far.mean(); self.g = g
        # token far-rate (smoothed)
        rows = []
        for q, f in zip(L.search_query.values, L.far.values):
            for t in set(toks(q)): rows.append((t, f))
        tf = pd.DataFrame(rows, columns=['t', 'f']).groupby('t').f.agg(['sum', 'count'])
        self.tok_far = ((tf['sum'] + 5 * g) / (tf['count'] + 5)).to_dict()
        qf = L.groupby('search_query').far.agg(['sum', 'count'])
        self.q_far = ((qf['sum'] + 2 * g) / (qf['count'] + 2)).to_dict()
        mf = L.groupby('mc').far.agg(['sum', 'count'])
        mcf = ((mf['sum'] + 10 * g) / (mf['count'] + 10))
        self.i_mcfar = pd.Series(mc[corpus]).map(mcf).fillna(g).values.astype(np.float32)
        txt = (items.item_title_raw.values[corpus] + ' ' + items.item_description_raw.values[corpus] + ' ' + items.item_infm_params_text.values[corpus])
        self.i_remote = np.array([len(REMOTE.findall(norm(s))) for s in txt], np.float32)
        tl = items.item_title_raw.values[corpus]
        self.i_remote_t = np.array([bool(REMOTE.search(norm(s))) for s in tl], np.int8)
        pt = items.item_infm_params_text.values[corpus]
        self.i_city = np.array(['По всему городу' in s for s in pt], np.int8)
        self.i_zones = np.array(['В выбранные зоны' in s for s in pt], np.int8)

    def add(self, F, qdf):
        qdf = qdf.reset_index(drop=True)
        qt = [toks(q) for q in qdf.search_query]
        tfar_mean = np.array([np.mean([self.tok_far.get(t, self.g) for t in ts]) if ts else self.g for ts in qt], np.float32)
        tfar_max = np.array([np.max([self.tok_far.get(t, self.g) for t in ts]) if ts else self.g for ts in qt], np.float32)
        qfar = np.array([self.q_far.get(q, np.nan) for q in qdf.search_query], np.float32)
        qremote = np.array([bool(REMOTE.search(norm(q))) for q in qdf.search_query], np.int8)
        qi = F.qi.values; ci = F.ci.values
        F['q_tfar_mean'] = tfar_mean[qi]; F['q_tfar_max'] = tfar_max[qi]; F['q_far'] = qfar[qi]; F['q_remote'] = qremote[qi]
        F['i_mcfar'] = self.i_mcfar[ci]; F['i_remote'] = self.i_remote[ci]; F['i_remote_t'] = self.i_remote_t[ci]
        F['i_city'] = self.i_city[ci]; F['i_zones'] = self.i_zones[ci]
        return F

# Вспомогательный контекст валидации (BM25-индексы, приор локации) для hyb_eval.py.
# Запуск из корня репозитория: python experiments/02_baselines/valctx.py (нужны data/ из scripts/prep.py … split.py)
"""Shared val context: corpus, rels, BM25 indexes, loc prior matrix (per unique search loc)."""
import sys, pickle, pandas as pd, numpy as np
sys.path.insert(0, 'src'); from common import *
class Ctx:
    def __init__(self, qdf, corpus, Tlog, fields=('title','par','desc','all')):
        self.items = pd.read_parquet('data/items.parquet', columns=['iid','item_location_id','item_latitude','item_longitude','item_category_id','item_microcat_id'])
        tk = pickle.load(open('data/item_toks.pkl','rb'))
        self.corpus = corpus; self.q = qdf.reset_index(drop=True)
        iloc_all = self.items.item_location_id.values; self.cl = iloc_all[corpus]
        self.idx = {}
        for f in fields:
            docs = [tk['title'][i]+tk['par'][i]+tk['desc'][i] for i in corpus] if f=='all' else [tk[f][i] for i in corpus]
            self.idx[f] = BM25(docs)
        T = Tlog.copy(); T['il'] = iloc_all[T.iid.values]
        self.pair = T.groupby(['search_location_id','il']).size(); self.N = T.groupby('search_location_id').size()
        self.g = T.il.value_counts(normalize=True)
        self.qt = [toks(q) for q in self.q.search_query]
        self.LP = {}
    def loc_logp(self, sloc, alpha=5.0):
        if sloc in self.LP: return self.LP[sloc]
        cl = pd.Series(self.cl); base = cl.map(self.g).fillna(1e-6).values
        Ns = self.N.get(sloc, 0)
        cnt = cl.map(self.pair[sloc]).fillna(0).values if Ns else np.zeros(len(cl))
        p = (cnt + alpha*base) / (Ns + alpha)
        if Ns == 0: p = np.where(self.cl==sloc, 0.77, p)
        self.LP[sloc] = np.log(p + 1e-7).astype(np.float32); return self.LP[sloc]

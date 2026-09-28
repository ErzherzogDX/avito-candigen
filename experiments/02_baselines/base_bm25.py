# Бейзлайн: BM25 по полям + приор локации (лучшее: все поля + локация, R@50 = 0.818).
# Запуск из корня репозитория: python experiments/02_baselines/base_bm25.py (нужны data/ из scripts/prep.py … split.py)
import sys, pickle, pandas as pd, numpy as np, time
sys.path.insert(0, 'src'); from common import *
items = pd.read_parquet('data/items.parquet', columns=['iid','item_location_id','item_infm_params_text'])
tk = pickle.load(open('data/item_toks.pkl','rb'))
T = pd.read_parquet('data/T_log.parquet'); val = pd.read_parquet('data/val_q.parquet'); corpus = np.load('data/val_corpus.npy')
iloc_all = items.item_location_id.values; cl = iloc_all[corpus]
t0=time.time()
idx = {f: BM25([tk[f][i] for i in corpus]) for f in ['title','par','desc']}
idx['all'] = BM25([tk['title'][i]+tk['par'][i]+tk['desc'][i] for i in corpus])
idx['tp'] = BM25([tk['title'][i]+tk['par'][i] for i in corpus])
print('index', time.time()-t0)
# location prior
T['il'] = iloc_all[T.iid.values]
pair = T.groupby(['search_location_id','il']).size()
N = T.groupby('search_location_id').size()
g = T.il.value_counts(normalize=True)
locs_c = pd.Series(cl)
def loc_logp(sloc, alpha=5.0):
    n = pair.get(sloc)
    Ns = N.get(sloc, 0)
    base = locs_c.map(g).fillna(1e-6).values
    cnt = locs_c.map(n).fillna(0).values if n is not None else np.zeros(len(cl))
    p = (cnt + alpha*base) / (Ns + alpha)
    if Ns == 0: p = np.where(cl==sloc, 0.77, p)
    return np.log(p + 1e-7)
qt = [toks(q) for q in val.search_query]
rels = [list(map(int, np.searchsorted(corpus, r))) for r in val.rel]   # rel in corpus-index space
LP = {s: loc_logp(s) for s in val.search_location_id.unique()}
print('prep', time.time()-t0)
def run(field, wloc, wf=None, K=50):
    preds=[]
    for q, s in zip(qt, val.search_location_id):
        if wf: sc = sum(w*idx[f].score(q) for f,w in wf.items())
        else: sc = idx[field].score(q)
        sc = sc + wloc*LP[s]
        preds.append(np.argpartition(-sc, K)[:K])
    return preds
for f in ['title','par','desc','all','tp']:
    for w in [0, 0.5, 1.0, 2.0]:
        print(f, w, round(recall_at(run(f, w), rels), 4))
for wf in [{'title':1,'par':0.5,'desc':0.3},{'title':1,'par':1,'desc':0.5},{'title':1,'par':0.3,'desc':0.2},{'all':1,'title':1}]:
    for w in [0.5, 1.0, 1.5]:
        print(wf, w, round(recall_at(run(None, w, wf), rels), 4))

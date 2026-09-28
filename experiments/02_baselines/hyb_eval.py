# Гибрид BM25 + zero-shot dense + локация: R@50 = 0.869.
# Запуск из корня репозитория: python experiments/02_baselines/hyb_eval.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd
sys.path.insert(0,'src'); from common import *; from valctx import Ctx
val = pd.read_parquet('data/val_q.parquet'); corpus = np.load('data/val_corpus.npy'); T = pd.read_parquet('data/T_log.parquet')
C = Ctx(val, corpus, T, fields=('all',))
rels = [list(map(int, np.searchsorted(corpus, r))) for r in val.rel]
D = np.load('data/zs_multilingual-e5-small_D.npy').astype(np.float32); Q = np.load('data/zs_multilingual-e5-small_Q.npy')
def ev(wd, wb, wl, K=50):
    preds=[]
    for i,(q,s) in enumerate(zip(C.qt, val.search_location_id)):
        sc = wl*C.loc_logp(s)
        if wb: b = C.idx['all'].score(q); sc = sc + wb*b/(b.max()+1e-6)
        if wd: sc = sc + wd*(D@Q[i])
        preds.append(np.argpartition(-sc, K)[:K])
    return round(recall_at(preds, rels),4)
for wl in [0.05, 0.1, 0.2]: print('dense+loc', wl, ev(1,0,wl), flush=True)
for wl in [0.1, 0.2, 0.3]: print('bm25n+loc', wl, ev(0,1,wl), flush=True)
for wd in [2, 5, 10]:
    for wl in [0.1, 0.2, 0.3]: print('hyb', wd, wl, ev(wd,1,wl), flush=True)

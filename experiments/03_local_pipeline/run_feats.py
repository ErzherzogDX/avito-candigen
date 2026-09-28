# Сборка кандидатов и признаков (src/pipeline.py) для val и запросов ранкера.
# Запуск из корня репозитория: python experiments/03_local_pipeline/run_feats.py (нужны data/ из scripts/prep.py … split.py)
"""Build candidate features for val and rk queries (val setting: corpus=V, log=T)."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, 'src'); from pipeline import Builder
tag = sys.argv[1]; dense_names = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] else []
nrk = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
knn_tag = sys.argv[4] if len(sys.argv) > 4 else None
corpus = np.load('data/val_corpus.npy'); T = pd.read_parquet('data/T_log.parquet')
val = pd.read_parquet('data/val_q.parquet'); rk = pd.read_parquet('data/rk_q.parquet').iloc[:nrk]
knn = (np.load('data/logQT_texts.npy', allow_pickle=True), np.load(f'data/emb_{knn_tag}_logQT.npy')) if knn_tag else None
B = Builder(corpus, T, knn=knn)
for name, qdf in [('val', val), ('rk', rk)]:
    dense = {}
    for dn in dense_names:
        D = np.load(f'data/emb_{dn}_valcorpus.npy'); Q = np.load(f'data/emb_{dn}_{name}Q.npy').astype(np.float32)
        dense[dn] = (Q, D.astype(np.float32))
    qplain = np.load(f'data/emb_{knn_tag}_{name}Qp.npy').astype(np.float32) if knn_tag else None
    F = B.build(qdf, dense, qplain=qplain)
    F.to_parquet(f'data/feats_{tag}_{name}.parquet', index=False)
    g = F.groupby('qi').y.sum(); nrel = qdf.rel.str.len().values
    print(name, 'rows', len(F), 'cands/q', len(F)/len(qdf), 'union recall', (g.reindex(range(len(qdf))).fillna(0).values / nrel).mean(), flush=True)

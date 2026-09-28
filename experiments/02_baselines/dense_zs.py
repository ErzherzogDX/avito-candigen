# Zero-shot dense (multilingual-e5-small): R@50 0.344 без локации.
# Запуск из корня репозитория: python experiments/02_baselines/dense_zs.py (нужны data/ из scripts/prep.py … split.py)
"""Zero-shot dense eval on val: encode val corpus + val queries, recall@50 with/without loc prior."""
import sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, 'src'); from common import recall_at
from sentence_transformers import SentenceTransformer
name, qp, dp, maxlen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tag = name.split('/')[-1]
val = pd.read_parquet('data/val_q.parquet'); corpus = np.load('data/val_corpus.npy')
it = pd.read_parquet('data/item_text.parquet').text.values
m = SentenceTransformer(name, device=('cuda' if torch.cuda.is_available() else 'mps')); m.max_seq_length = maxlen
t0 = time.time()
D = m.encode([dp + it[i] for i in corpus], batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
print('enc docs', time.time()-t0, flush=True)
Q = m.encode([qp + q for q in val.search_query], batch_size=256, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
np.save(f'data/zs_{tag}_D.npy', D); np.save(f'data/zs_{tag}_Q.npy', Q)
rels = [list(map(int, np.searchsorted(corpus, r))) for r in val.rel]
S = Q @ D.astype(np.float32).T
preds = [np.argpartition(-s, 50)[:50] for s in S]
print(tag, 'dense R@50 no loc', round(recall_at(preds, rels), 4), flush=True)

# Кодирование корпуса/запросов дообученным энкодером (mean pooling).
# Запуск из корня репозитория: python experiments/03_local_pipeline/encode.py (нужны data/ из scripts/prep.py … split.py)
"""Encode sets with a (fine-tuned) mean-pooling encoder.
usage: encode.py MODEL_DIR TAG sets(comma: valcorpus,valQ,rkQ,benchcorpus,benchQ,logQ) [qprefix] [dprefix] [nrk]"""
import sys, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
mdir, tag, sets = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
qp = sys.argv[4] if len(sys.argv) > 4 else 'query: '; dp = sys.argv[5] if len(sys.argv) > 5 else 'passage: '
nrk = int(sys.argv[6]) if len(sys.argv) > 6 else 30000
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
tok = AutoTokenizer.from_pretrained(mdir); model = AutoModel.from_pretrained(mdir).to(dev).eval()
@torch.no_grad()
def enc(texts, L, bs):
    out = []
    order = np.argsort([len(t) for t in texts])
    for i in range(0, len(texts), bs):
        b = tok([texts[j] for j in order[i:i + bs]], padding=True, truncation=True, max_length=L, return_tensors='pt').to(dev)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            h = model(**b).last_hidden_state
        h = h.float(); m = b['attention_mask'].unsqueeze(-1).float()
        out.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy().astype(np.float16))
    E = np.concatenate(out); R = np.empty_like(E); R[order] = E
    return R
def qtext(q, f): return q if not f else f'{q} | {f}'
it = None
for s in sets:
    t0 = time.time()
    if s.endswith('corpus'):
        if it is None: it = pd.read_parquet('data/item_text.parquet').text.values
        if s == 'valcorpus': corpus = np.load('data/val_corpus.npy')
        else:
            items = pd.read_parquet('data/items.parquet', columns=['iid', 'in_corpus'])
            corpus = np.sort(items.iid.values[items.in_corpus.values == 1])
        E = enc([dp + it[i] for i in corpus], 128, 128)
    elif s in ('logQT', 'logQF'):
        src = 'data/T_log.parquet' if s == 'logQT' else 'data/log.parquet'
        texts = np.array(sorted(pd.read_parquet(src, columns=['search_query']).search_query.unique()), dtype=object)
        np.save(f'data/{s}_texts.npy', texts)
        E = enc([qp + t for t in texts], 32, 512)
    elif s.endswith('Qp'):
        src = {'valQp': 'data/val_q.parquet', 'rkQp': 'data/rk_q.parquet', 'benchQp': 'data/bench_q.parquet'}[s]
        q = pd.read_parquet(src)
        if s == 'rkQp': q = q.iloc[:nrk]
        E = enc([qp + t for t in q.search_query], 32, 512)
    else:
        if s == 'valQ': q = pd.read_parquet('data/val_q.parquet')
        elif s == 'rkQ': q = pd.read_parquet('data/rk_q.parquet').iloc[:nrk]
        elif s == 'benchQ': q = pd.read_parquet('data/bench_q.parquet')
        E = enc([qp + qtext(a, b) for a, b in zip(q.search_query, q.search_infm_params_text)], 32, 512)
    np.save(f'data/emb_{tag}_{s}.npy', E)
    print(s, E.shape, f'{time.time() - t0:.0f}s', flush=True)

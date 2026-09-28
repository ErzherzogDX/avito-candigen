# Дообучение bi-encoder (e5-small) на парах «запрос -> выбор»: InfoNCE, in-batch негативы, батч без дублей. Им обучены e5s_T/e5s_F финала.
# Запуск из корня репозитория: python experiments/03_local_pipeline/ft_biencoder.py (нужны data/ из scripts/prep.py … split.py)
"""Fine-tune a bi-encoder (mean pooling) on query->chosen item pairs with in-batch negatives (+ optional hard negs).
usage: ft_biencoder.py MODEL LOG_PARQUET OUT_DIR [epochs] [bs] [lr] [qprefix] [dprefix]"""
import sys, math, random, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
name, logp, out = sys.argv[1], sys.argv[2], sys.argv[3]
epochs = float(sys.argv[4]) if len(sys.argv)>4 else 1; bs = int(sys.argv[5]) if len(sys.argv)>5 else 128
lr = float(sys.argv[6]) if len(sys.argv)>6 else 5e-5
qp = sys.argv[7] if len(sys.argv)>7 else 'query: '; dp = sys.argv[8] if len(sys.argv)>8 else 'passage: '
QLEN, DLEN, CAP, SCALE = 32, 128, 20, 20.0
torch.manual_seed(0); random.seed(0); np.random.seed(0)
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
it = pd.read_parquet('data/item_text.parquet').text.values
log = pd.read_parquet(logp)
def qtext(q, f): return q if not f else f'{q} | {f}'
log['qt'] = [qtext(q, f) for q, f in zip(log.search_query, log.search_infm_params_text)]
pairs = log[['qt','search_query','iid']].drop_duplicates(['qt','iid'])
pairs = pairs.sample(frac=1, random_state=0).groupby('search_query').head(CAP).reset_index(drop=True)
print('pairs', len(pairs), flush=True)
tok = AutoTokenizer.from_pretrained(name); model = AutoModel.from_pretrained(name).to(dev); model.train()
def enc(texts, L):
    # static shapes help MPS; dynamic padding + bf16 autocast is faster on CUDA
    b = tok(texts, padding='max_length' if dev == 'mps' else True, truncation=True, max_length=L, return_tensors='pt').to(dev)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
        h = model(**b).last_hidden_state
    h = h.float(); m = b['attention_mask'].unsqueeze(-1).float()
    return F.normalize((h*m).sum(1)/m.sum(1), dim=-1)
def batches():
    idx = np.random.permutation(len(pairs)); cur, seenq, seend = [], set(), set()
    for i in idx:
        q, it_ = pairs.search_query.iat[i], pairs.iid.iat[i]
        if q in seenq or it_ in seend: continue
        cur.append(i); seenq.add(q); seend.add(it_)
        if len(cur) == bs: yield cur; cur, seenq, seend = [], set(), set()
nb = int(len(pairs)/bs*epochs)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s/(0.05*nb)) * max(0.0, (nb-s)/(nb*0.95)))
step, t0, lsum = 0, time.time(), 0
while step < nb:
    for b in batches():
        qs = [qp + pairs.qt.iat[i] for i in b]; ds = [dp + it[pairs.iid.iat[i]] for i in b]
        qe, de = enc(qs, QLEN), enc(ds, DLEN)
        s = qe @ de.T * SCALE
        lab = torch.arange(len(b), device=dev)
        loss = (F.cross_entropy(s, lab) + F.cross_entropy(s.T, lab)) / 2
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        step += 1; lsum += loss.item()
        if step % 100 == 0: print(f'step {step}/{nb} loss {lsum/100:.4f} {time.time()-t0:.0f}s', flush=True); lsum = 0
        if step >= nb: break
model.save_pretrained(out); tok.save_pretrained(out)
print('saved', out, flush=True)

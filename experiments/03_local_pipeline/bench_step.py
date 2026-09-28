# Замер скорости шага обучения на MPS (M4) для выбора батча.
# Запуск из корня репозитория: python experiments/03_local_pipeline/bench_step.py (нужны data/ из scripts/prep.py … split.py)
import time, torch, torch.nn.functional as F, sys
from transformers import AutoTokenizer, AutoModel
name = sys.argv[1]; bs = int(sys.argv[2]); pad = sys.argv[3]; amp = sys.argv[4]=='1'; DL=int(sys.argv[5]) if len(sys.argv)>5 else 128
dev='mps'; tok=AutoTokenizer.from_pretrained(name); model=AutoModel.from_pretrained(name).to(dev); model.train()
[p.requires_grad_(False) for p in model.embeddings.parameters()] if len(sys.argv)>6 else None; opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
q=['query: ремонт стиральных машин на дому']*bs; d=['passage: Ремонт стиральных машин. Быстро качественно с гарантией выезд мастера на дом '*10]*bs
def enc(t,L):
    b=tok(t,padding=pad if pad!='dyn' else True,truncation=True,max_length=L,return_tensors='pt').to(dev)
    with torch.autocast('mps', dtype=torch.float16, enabled=amp):
        h=model(**b).last_hidden_state
    m=b['attention_mask'].unsqueeze(-1).float(); return F.normalize((h.float()*m).sum(1)/m.sum(1),dim=-1)
for i in range(8):
    t=time.time(); s=enc(q,32)@enc(d,DL).T*20; loss=F.cross_entropy(s,torch.arange(bs,device=dev)); opt.zero_grad(); loss.backward(); opt.step(); torch.mps.synchronize()
    print(i, round(time.time()-t,3), flush=True)

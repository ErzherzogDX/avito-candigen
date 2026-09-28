# Адаптация основного энкодера к корпусу без разметки: псевдо-пары «заголовок -> объявление без заголовка» + реплей реальных пар.
# Запуск в том же ноутбуке после секции 8 (или хотя бы секций 0–5 + наличия *_F моделей), в новой ячейке:   %run -i experiments/06_notebook_addons/adapt.py
# Шаги: (A) T-модель адаптируется на val-корпусе -> проверка на валидации (dense и полный пайплайн);
#       (B) если валидация не ухудшилась — F-модель адаптируется на корпусе бенчмарка -> subs/sub_adapt.csv.
from common import toks_raw

# параметры можно задать заранее в ячейке перед %run -i experiments/06_notebook_addons/adapt.py
N_PSEUDO = globals().get('N_PSEUDO', 600 if SMOKE else 60000)   # псевдо-пар из корпуса (1 эпоха по ним; столько же батчей реальных пар)
AD_LR = globals().get('AD_LR', 1e-5)                            # меньше базового: продолжаем обучение уже хорошей модели
GATE_DENSE = globals().get('GATE_DENSE', -0.005)                # стоп, если dense dR@35wloc упал сильнее
GATE_PIPE = globals().get('GATE_PIPE', -0.001)                  # на бенчмарк — только если R@35w не ухудшился сильнее
FORCE_BENCH = globals().get('FORCE_BENCH', False)               # True: собрать сабмит независимо от валидации

P = PIPE[PBEST]; (KEY, HN) = P['encs'][0]
T_DIR = enc_ref(KEY, HN)[1]; T_TAG = P['tags'][0]
F_DIR = f'models/{TAG}{KEY}_F' + ('_hn' if HN else ''); F_TAG = f'{KEY}F' + ('hn' if HN else '')
D2 = P['encs'][1] if len(P['encs']) > 1 else None
print(f'adapt: pipeline {PBEST}, main encoder {KEY} (hard-neg={HN}); T={T_DIR}, F={F_DIR}', flush=True)

_it = pd.read_parquet('data/items.parquet', columns=['item_title_raw', 'item_microcat_id'])
TITLES, MCAT = _it.item_title_raw.values, _it.item_microcat_id.values; del _it
def notitle(i):
    t = ITEXT[i]; return t[len(TITLES[i]) + 2:] if t.startswith(TITLES[i]) else t

def make_pseudo(corpus, n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'q': [' '.join(toks_raw(TITLES[i])[:6]) for i in corpus], 'iid': corpus})
    df = df[df.q.str.len() >= 3].sample(frac=1, random_state=seed).drop_duplicates('q').head(n).reset_index(drop=True)
    by_mc = pd.Series(corpus).groupby(MCAT[corpus]).apply(np.array).to_dict()
    hn = np.empty(len(df), np.int64)
    for k, i in enumerate(df.iid.values):
        pool = by_mc[MCAT[i]]
        for _ in range(5):   # негатив из той же микрокатегории, с другим текстом
            j = pool[rng.integers(len(pool))] if len(pool) > 1 else corpus[rng.integers(len(corpus))]
            if j != i and notitle(j) != notitle(i): break
        else:
            j = corpus[rng.integers(len(corpus))]
        hn[k] = j
    df['hn'] = hn
    return df

def train_adapt(key, init_dir, out_dir, sup, sup_hn, pseudo, lr=AD_LR):
    if prepare_out(out_dir): return out_dir
    cfg = ENCODERS[key]; enc = Encoder(cfg, init_dir); model = enc.model; model.train()
    if cfg['gc']:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, 'use_cache'): model.config.use_cache = False
    bs = cfg['bs']; nb = 2 * max(1, len(pseudo) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / max(1, 0.05 * nb)) * max(0.0, (nb - s) / max(1, 0.95 * nb)))
    rng = np.random.default_rng(0)
    S = dict(q=np.array(enc.q(sup.qt.tolist()), dtype=object), k=sup.q.values, i=sup.iid.values, h=np.asarray(sup_hn))
    Ps = dict(q=np.array(enc.q(pseudo.q.tolist()), dtype=object), k=pseudo.q.values, i=pseudo.iid.values, h=pseudo.hn.values)
    def batches(D):
        while True:
            cur, sq, sd = [], set(), set()
            for j in rng.permutation(len(D['i'])):
                if D['k'][j] in sq or D['i'][j] in sd or D['h'][j] in sd or D['h'][j] == D['i'][j]: continue
                cur.append(j); sq.add(D['k'][j]); sd.add(D['i'][j]); sd.add(D['h'][j])
                if len(cur) == bs: yield cur; cur, sq, sd = [], set(), set()
    g_sup, g_ps = batches(S), batches(Ps); t0, lsum = time.time(), [0.0, 0.0]
    for step in range(nb):
        real = step % 2 == 0          # батчи не смешиваем: у псевдо-документов нет заголовка
        D = S if real else Ps; b = next(g_sup if real else g_ps)
        ids = list(D['i'][b]) + list(D['h'][b])
        docs = enc.d(ids) if real else [cfg['dp'] + notitle(i) for i in ids]
        qe = enc.embed(list(D['q'][b]), cfg['qlen']); de = enc.embed(docs, DLEN)
        s = qe @ de.T * SCALE; lab = torch.arange(len(b), device=DEV)
        loss = (F.cross_entropy(s, lab) + F.cross_entropy(s[:, :len(b)].T, lab)) / 2
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        lsum[0 if real else 1] += loss.item()
        if (step + 1) % 100 == 0 or step + 1 == nb:
            print(f'  adapt {key} step {step + 1}/{nb} loss real {lsum[0] / 50:.4f} pseudo {lsum[1] / 50:.4f} {time.time() - t0:.0f}s', flush=True); lsum = [0.0, 0.0]
    save_bf16(model, enc.tok, out_dir); enc.free()
    return out_dir

# ---------- (A) валидация: T-модель адаптируется на val-корпусе (без его разметки)
PS_V = make_pseudo(VCORP, N_PSEUDO, seed=0); print('pseudo pairs (val corpus):', len(PS_V), PS_V.q.head(3).tolist(), flush=True)
T_AD = train_adapt(KEY, T_DIR, T_DIR + '_ad', PAIRS_T, pick_hardneg(MINED_T, TCORP, seed=1), PS_V)
TA_TAG = T_TAG + 'ad'
encode_sets(KEY, T_AD, TA_TAG, ['valcorpus', 'valQ'])
base_d, ad_d = dense_eval(T_TAG), dense_eval(TA_TAG)
log_result('adapt-dense', f'{T_TAG} (база)', base_d); log_result('adapt-dense', TA_TAG, ad_d)
go = FORCE_BENCH or ad_d['dR@35wloc'] >= base_d['dR@35wloc'] + GATE_DENSE
if not go: print('СТОП: адаптация ухудшила dense-поиск, дальше не идём', flush=True)

if go:
    AD_NAME = f'{PBEST}_ad'
    pipe_encode(KEY, TA_TAG, T_AD)
    Fv, Fr = build_pipeline_feats(AD_NAME, TA_TAG, P['tags'][1] if D2 else None); Fv, Fr = add_ranks(Fv), add_ranks(Fr)
    feats_ad = feat_cols(Fr); ms_ad = lgb_fit(Fr, feats_ad)
    M_AD = eval_rank(Fv, lgb_pred(ms_ad, Fv, feats_ad)); log_result('adapt-pipeline', AD_NAME, M_AD)
    base_R = PIPE[PBEST]['metric']['R@35w']; delta = M_AD['R@35w'] - base_R
    print(f'R@35w: база {base_R:.4f} -> адаптация {M_AD["R@35w"]:.4f} ({delta * 100:+.2f} п.п.)', flush=True)
    go = FORCE_BENCH or delta >= GATE_PIPE
    if not go: print('СТОП: полный пайплайн на валидации ухудшился, сабмит не собираем', flush=True)

# ---------- (B) бенчмарк: F-модель адаптируется на корпусе бенчмарка
if go:
    mined_F = pickle.load(open(f'data/{TAG}mined_F_{KEY}.pkl', 'rb')) if Path(f'data/{TAG}mined_F_{KEY}.pkl').exists() \
        else mine(KEY, F_DIR, PAIRS_F, LOG, FCORP, LP_F, cache=f'data/{TAG}mined_F_{KEY}.pkl')
    PS_B = make_pseudo(BCORP, N_PSEUDO, seed=0); print('pseudo pairs (bench corpus):', len(PS_B), PS_B.q.head(3).tolist(), flush=True)
    F_AD = train_adapt(KEY, F_DIR, F_DIR + '_ad', PAIRS_F, pick_hardneg(mined_F, FCORP, seed=1), PS_B)
    FA_TAG = F_TAG + 'ad'; pipe_encode(KEY, FA_TAG, F_AD, bench=True)
    tf2 = None
    if D2:
        k2, hn2 = D2; tf2 = f'{k2}F' + ('hn' if hn2 else '')
        pipe_encode(k2, tf2, f'models/{TAG}{k2}_F' + ('_hn' if hn2 else ''), bench=True)
    fb = f'data/{TAG}feats_bench_{AD_NAME}.parquet'
    if Path(fb).exists(): FBa = pd.read_parquet(fb)
    else:
        B = Builder(BCORP, LOG, knn=(np.load('data/logQF_texts.npy', allow_pickle=True), np.load(emb_path(FA_TAG, 'logQF'))))
        dense = {'d1': load_dense(FA_TAG, 'bench')}
        if tf2: dense['d2'] = load_dense(tf2, 'bench')
        FBa = par_build(B, BQ, dense, np.load(emb_path(FA_TAG, 'benchQp')).astype(np.float32), labels=False)
        FBa.to_parquet(fb, index=False); del B; gc.collect()
    FBa = add_ranks(FBa)
    ms1a = lgb_fit(pd.concat([Fr, Fv.assign(qi=Fv.qi + 10 ** 7)], ignore_index=True), feats_ad)
    FBa['s'] = lgb_pred(ms1a, FBa, feats_ad)
    if 'finalize' not in globals():
        raise RuntimeError('нет функции finalize: выполните сначала раздел 8 ноутбука')
    finalize(FBa, 's', f'subs/{TAG}sub_adapt.csv')
    print('готово: subs/sub_adapt.csv', flush=True)

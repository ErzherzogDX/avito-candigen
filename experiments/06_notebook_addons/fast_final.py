# Быстрый финал (запасной путь под дедлайн): бенчмарк кодируется T-версиями энкодеров, без переобучения на всём train.
# Запуск в том же ноутбуке ПОСЛЕ секции 5 (лучше — после секции 6), в новой ячейке:   %run -i experiments/06_notebook_addons/fast_final.py
# Пишет subs/sub_fast_stage1.csv (без cross-encoder) и, если секция 6 прошла, subs/sub_fast.csv (со вторым проходом).
P = PIPE[PBEST]
if 'FEATS1' not in globals():   # секция 6 не выполнялась
    _fr = pd.read_parquet(f'data/{TAG}feats_{PBEST}_rk.parquet'); FEATS1 = feat_cols(add_ranks(_fr)); del _fr
# cross-encoder берём, только если секция 6 дошла до конца (есть USE_CE и CE-признаки у обучающих строк)
_ce_ready = all(n in globals() for n in ('USE_CE', 'TR2', 'VA2', 'FEATS2', 'CE_T')) and 'ce_z' in globals()['TR2'].columns
USE_CE = bool(globals().get('USE_CE', False)) and _ce_ready
print('fast final:', PBEST, P['encs'], '| cross-encoder:', USE_CE, flush=True)

FTAGS = []
for key, hn in P['encs']:
    t, d = enc_ref(key, hn); pipe_encode(key, t, d, bench=True); FTAGS.append(t)   # T-модели кодируют корпус бенчмарка
tf1, tf2 = FTAGS[0], (FTAGS[1] if len(FTAGS) > 1 else None)

fb = f'data/{TAG}feats_bench_fast_{PBEST}.parquet'
if Path(fb).exists(): FB = pd.read_parquet(fb)
else:
    B = Builder(BCORP, LOG, knn=(np.load('data/logQF_texts.npy', allow_pickle=True), np.load(emb_path(tf1, 'logQF'))))
    dense = {'d1': load_dense(tf1, 'bench')}
    if tf2: dense['d2'] = load_dense(tf2, 'bench')
    FB = par_build(B, BQ, dense, np.load(emb_path(tf1, 'benchQp')).astype(np.float32), labels=False)
    FB.to_parquet(fb, index=False); del B; gc.collect()
FB = add_ranks(FB)

Fv, Fr = build_pipeline_feats(PBEST, *tags_of(P)); Fv, Fr = add_ranks(Fv), add_ranks(Fr)
ms1 = lgb_fit(pd.concat([Fr, Fv.assign(qi=Fv.qi + 10 ** 7)], ignore_index=True), FEATS1)
FB['s'] = lgb_pred(ms1, FB, FEATS1); del Fv, Fr; gc.collect()

def finalize(Fb, score_col, out_csv):
    Fb = Fb.copy(); qcat = BQ.search_category.values[Fb.qi.values]
    Fb = Fb[~((qcat == 114) & (Fb.i_cat114.values == 0))].sort_values(['qi', score_col], ascending=[True, False])
    qcat = BQ.search_category.values[Fb.qi.values]
    non = Fb[(qcat == 0) & (Fb.i_cat114.values == 0)]
    good = (non.bmn_all >= 0.4) | (non.rk_bm_all <= 20)
    for c in [c for c in Fb.columns if c.startswith('rk_dn_')]: good |= non[c] <= 20
    res_non = non[good].groupby('qi').head(R0)
    rest = Fb.drop(res_non.index)
    k_rest = (50 - res_non.groupby('qi').size()).reindex(range(len(BQ))).fillna(50).astype(int)
    rest = rest[rest.groupby('qi').cumcount() < k_rest.values[rest.qi.values]]
    top = pd.concat([res_non, rest]).sort_values(['qi', score_col], ascending=[True, False])
    ids = ITEMS.item_id.values
    ans = top.groupby('qi').ci.apply(lambda c: ' '.join(ids[BCORP[c.values]]))
    out = pd.DataFrame({'query_id': BQ.query_id.values, 'answer': ans.reindex(range(len(BQ))).fillna('').values})
    out.to_csv(out_csv, index=False); print('written', out_csv, out.shape, '| cat-0 queries with reserved non-114:', res_non.qi.nunique(), flush=True)
    if not SMOKE: run(f'{sys.executable} scripts/check_answer.py {out_csv}')
    else: assert (out.answer.str.split().str.len() == 50).all(), 'answer shorter than 50'

finalize(FB, 's', f'subs/{TAG}sub_fast_stage1.csv')

if USE_CE:
    ms2 = lgb_fit(pd.concat([TR2, VA2.assign(qi=VA2.qi + 10 ** 7)], ignore_index=True), FEATS2)
    FB2 = topk_rows(FB, FB.s.values, CE_TOPK)
    FB2 = ce_feats(FB2, ce_score(CE_T, [BQ_QT[i] for i in FB2.qi], BCORP[FB2.ci.values]))
    FB2['s2'] = lgb_pred(ms2, FB2, FEATS2)
    FBc = FB.merge(FB2[['qi', 'ci', 's2']], on=['qi', 'ci'], how='left'); FBc['s2'] = FBc.s2.fillna(FBc.s - 1e6)
    finalize(FBc, 's2', f'subs/{TAG}sub_fast.csv')

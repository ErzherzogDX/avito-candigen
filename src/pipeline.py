"""Candidate generation + feature building, shared by val and bench.
build(qdf, corpus, log, dense) -> DataFrame(qi, ci, features...)
  qdf: queries (search_query, search_location_id, search_infm_params_text, search_category, [rel])
  corpus: np.array of global iids
  log: query log used as 'history' (search_*, iid) — T-log for val, full train for bench
  dense: dict name -> (Q [nq,d] float32, D [ncorpus,d] float16)

Схема для одного запроса:
  1) кандидатогенерация — объединение нескольких ретриверов (BM25 с приором локации и без, BM25 по заголовку,
     dense-поиск с локацией и глобально, гибрид dense+BM25+локация, символьные 3-граммы, вероятные микрокатегории,
     «похожие запросы из лога»), итого ~500 кандидатов;
  2) для каждого кандидата — ~60 признаков (текст, dense, локация и расстояние, фильтры, приоры микрокатегорий, история, свойства объявления),
     по которым дальше учится LightGBM LambdaRank (в ноутбуке).
Один и тот же код используется для валидации (корпус V, лог T) и для бенчмарка (корпус бенчмарка, весь train) —
поэтому признаки на обучении ранкера и на инференсе считаются одинаково.
"""
import sys, pickle, time, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, 'src')
from common import *

ITEMS = None
def items_df():
    """Ленивая загрузка таблицы объявлений (одна на процесс; общая для всех Builder)."""
    global ITEMS
    if ITEMS is None:
        ITEMS = pd.read_parquet('data/items.parquet', columns=['iid', 'item_location_id', 'item_latitude', 'item_longitude',
            'item_category_id', 'item_microcat_id', 'item_rating', 'item_rating_reviews_count', 'item_price',
            'item_is_phone_hidden', 'item_is_message_forbidden', 'item_infm_params_text', 'item_title_raw', 'item_description_raw'])
    return ITEMS

# Сколько кандидатов берёт каждый ретривер (подобрано на валидации: потолок полноты кандидатов ~0.977 при ~520 кандидатах на запрос)
#   k_bm_loc   — BM25 по всем полям + приор локации        k_title_loc — BM25 по заголовку + приор локации
#   k_bm_glob  — BM25 без локации (онлайн-услуги)          k_dn_loc / k_dn_glob — dense с локацией / глобально
#   k_hyb      — гибрид dense + BM25 + локация            k_chr — символьные 3-граммы (опечатки)
#   k_mc       — вероятные микрокатегории и kNN по логу    wl_bm — вес приора локации в BM25; alpha_loc — сглаживание приора
CFG = dict(k_bm_loc=200, k_hyb=300, k_bm_glob=40, k_dn_glob=40, k_dn_loc=150, k_title_loc=80, k_chr=60, k_mc=60, wl_bm=0.1, alpha_loc=5.0)
# Счётные признаки (частота запроса, история объявления, счётчики локаций) растут с размером лога.
# Ранкер обучается на признаках от T-лога (251 306 строк), а на бенчмарке история — весь train (~2x),
# поэтому все счётчики нормируются к размеру T-лога: scale = REF_LOG / len(log).
REF_LOG = 251306
REF_STD = 0.108   # per-query std of cosine sims of the fine-tuned e5-small; other encoders are rescaled to it for candidate weights


class Builder:
    """Индексы и статистики по корпусу и логу; build() собирает кандидатов и признаки для набора запросов."""
    def __init__(self, corpus, log, cfg=CFG, knn=None):
        """knn: (texts array of unique log queries, their plain-text embeddings float16) or None"""
        t0 = time.time()
        self.cfg = cfg
        it = items_df(); tk = pickle.load(open('data/item_toks.pkl', 'rb'))
        self.corpus = corpus; n = len(corpus)
        self.cl = it.item_location_id.values[corpus]           # локация каждого объявления корпуса
        # --- BM25 по полям: заголовок, параметры услуги, описание и «всё вместе»
        self.idx = {'title': BM25([tk['title'][i] for i in corpus]),
                    'par': BM25([tk['par'][i] for i in corpus]),
                    'desc': BM25([tk['desc'][i] for i in corpus]),
                    'all': BM25([tk['title'][i] + tk['par'][i] + tk['desc'][i] for i in corpus])}
        # --- символьные 3-граммы TF-IDF по заголовку и названию услуги: ловят опечатки («иласос», «первозка»)
        from sklearn.feature_extraction.text import TfidfVectorizer
        sub = it.iloc[corpus]
        ctext = [norm(t) + ' ' + norm(param_service_text(p))[:200] for t, p in zip(sub.item_title_raw.values, sub.item_infm_params_text.values)]
        self.chv = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, sublinear_tf=True, dtype=np.float32)
        self.CH = self.chv.fit_transform(ctext).tocsr()
        # item static features
        # (i_cat114 — категория «Услуги»; в ранкер не идёт: в train почти нет объявлений других категорий, используется только правилами)
        self.stat = pd.DataFrame({
            'i_cat114': (sub.item_category_id.values == 114).astype(np.int8),
            'i_rating': sub.item_rating.values, 'i_reviews': np.log1p(sub.item_rating_reviews_count.fillna(0).values),
            'i_price': np.log1p(sub.item_price.clip(lower=0).fillna(-1).values + 1),
            'i_phone_hidden': sub.item_is_phone_hidden.astype(float).values, 'i_msg_forb': sub.item_is_message_forbidden.astype(float).values,
            'i_desc_len': np.log1p(sub.item_description_raw.str.len().values), 'i_title_len': sub.item_title_raw.str.len().values,
        })
        self.mc = sub.item_microcat_id.values
        # параметры объявления для сверки с фильтрами поиска («Вид услуги» совпадает в 98.5% пар train, «Тип услуги» — в 94.6%)
        pd_ = [param_dict(s) for s in sub.item_infm_params_text.values]
        self.i_vid = np.array([(d.get('Вид услуги') or [''])[0] for d in pd_], dtype=object)
        self.i_tip = [set(d.get('Тип услуги', [])) for d in pd_]
        self.i_pd = pd_
        self.lat = sub.item_latitude.values; self.lon = sub.item_longitude.values
        # ---- log-derived stats
        L = log.copy(); self.scale = REF_LOG / len(L)
        L['il'] = it.item_location_id.values[L.iid.values]
        L['mc'] = it.item_microcat_id.values[L.iid.values]
        # Приор локации P(локация объявления | локация поиска). 83% пар train — одна и та же локация, но поиск по региону
        # (например, 107620 «Москва и область») ведёт в город, а часть услуг выбирают издалека — поэтому приор, а не жёсткий фильтр.
        self.pair = L.groupby(['search_location_id', 'il']).size()
        self.N = L.groupby('search_location_id').size()
        self.g = L.il.value_counts(normalize=True)                      # глобальное распределение — для сглаживания
        self.cl_base = pd.Series(self.cl).map(self.g).fillna(1e-6).values
        # search-loc centroid: median coords of items chosen from that sloc
        L['lat'] = it.item_latitude.values[L.iid.values]; L['lon'] = it.item_longitude.values[L.iid.values]
        self.cent = L.groupby('search_location_id')[['lat', 'lon']].median()
        itl = pd.DataFrame({'l': it.item_location_id.values, 'lat': it.item_latitude.values, 'lon': it.item_longitude.values}).groupby('l').median()
        self.cent_item = itl                                            # запасной центр локации (если её нет в логе)
        # microcat priors: exact query text, and token-level
        #   точный текст: P(микрокатегория | запрос) для запросов, встречавшихся в логе (37% бенчмарка);
        #   по токенам: наивный Байес «слово запроса -> микрокатегория» — работает и для новых запросов.
        self.mc_list = np.unique(np.concatenate([L.mc.values, self.mc]))
        self.mc_pos = {m: j for j, m in enumerate(self.mc_list)}
        self.q_mc = L.groupby(['search_query', 'mc']).size()
        self.q_n = L.groupby('search_query').size()
        qtok = [set(toks(q)) for q in L.search_query.values]
        vocab = {}
        rows, cols = [], []
        for r, (ts, m) in enumerate(zip(qtok, L.mc.values)):
            for t in ts:
                j = vocab.setdefault(t, len(vocab)); rows.append(j); cols.append(self.mc_pos[m])
        M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(vocab), len(self.mc_list)))
        M.sum_duplicates()
        self.tok_vocab = vocab; self.tok_mc = M
        self.tok_n = np.asarray(M.sum(1)).ravel()
        prior = np.bincount([self.mc_pos[m] for m in L.mc.values], minlength=len(self.mc_list)).astype(np.float32)
        self.mc_prior = (prior + 1) / (prior.sum() + len(prior))
        self.cmc = np.array([self.mc_pos[m] for m in self.mc])
        # item history: queries that chose the item (only items in corpus)
        # (~9.6% объявлений корпуса бенчмарка уже выбирали в train — их прошлые запросы сильный сигнал)
        cpos = pd.Series(np.arange(n), index=corpus)
        Lc = L[L.iid.isin(corpus)]
        self.hist_n = np.zeros(n, np.float32)
        hc = Lc.iid.value_counts(); self.hist_n[cpos[hc.index].values] = hc.values
        self.hist_toks = {}; self.hist_q = {}
        for iid, grp in Lc.groupby('iid').search_query:
            c = cpos[iid]; self.hist_q[c] = set(grp.values)
            self.hist_toks[c] = set(t for q in grp.values for t in toks(q))
        # --- «похожие запросы из лога» (kNN по эмбеддингам запросов): для нового запроса находим ~20 семантически близких
        #     запросов train и переносим их опыт — распределение микрокатегорий и слова заголовков выбранных объявлений (PRF).
        self.knn = None
        if knn is not None:
            ktexts, KE = knn
            kpos = pd.Series(np.arange(len(ktexts)), index=ktexts)
            L2 = L[L.search_query.isin(kpos.index)]
            ui = kpos[L2.search_query.values].values
            # per-query-text microcat distribution
            M = sp.csr_matrix((np.ones(len(L2), np.float32), (ui, [self.mc_pos[m] for m in L2.mc.values])), shape=(len(ktexts), len(self.mc_list)))
            M.sum_duplicates(); M = sp.diags(1 / np.maximum(np.asarray(M.sum(1)).ravel(), 1)) @ M
            # per-query-text title-stem distribution (in title-BM25 vocab of this corpus)
            voc = self.idx['title'].vocab; rows, cols = [], []
            for u, iid in zip(ui, L2.iid.values):
                for t in set(tk['title'][iid]):
                    j = voc.get(t)
                    if j is not None: rows.append(u); cols.append(j)
            TT = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(ktexts), len(voc)))
            TT.sum_duplicates()
            cnt = np.bincount(ui, minlength=len(ktexts)).astype(np.float32)
            TT = sp.diags(1 / np.maximum(cnt, 1)) @ TT
            self.knn = dict(E=KE.astype(np.float32), MC=M.tocsr(), TT=TT.tocsr(), texts=ktexts)
        print(f'Builder ready {time.time()-t0:.0f}s', flush=True)

    def knn_feats(self, qe, q, c, K=20, tau=0.05):
        """K ближайших запросов лога с весами softmax(sim/tau) -> (микрокатегории, PRF-скор по заголовкам, max sim, mean sim)."""
        kn = self.knn; sims = kn['E'] @ qe
        nb = np.argpartition(-sims, K)[:K]
        s_ = sims[nb]; w = np.exp((s_ - s_.max()) / tau); w /= w.sum()
        mcd = np.asarray(w @ kn['MC'][nb].toarray()).ravel()
        bag = np.asarray(w @ kn['TT'][nb].toarray()).ravel()
        top = np.argpartition(-bag, 30)[:30]; top = top[bag[top] > 0.02]     # «расширенный запрос»: до 30 самых весомых слов заголовков
        W = self.idx['title'].W
        prf = np.asarray(W[:, top] @ bag[top]).ravel() if len(top) else np.zeros(W.shape[0], np.float32)
        return mcd[self.cmc], prf, float(s_.max()), float(s_.mean())

    def cov(self, field, qts, c):
        """Сколько разных слов запроса встречается в поле кандидата."""
        ids = self.idx[field].qvec(qts)
        if not ids: return np.zeros(len(c), np.float32)
        return np.asarray((self.idx[field].W[:, ids].tocsr()[c] > 0).sum(1)).ravel().astype(np.float32)

    def loc_logp(self, sloc):
        """log P(локация объявления | локация поиска) для всего корпуса (сглаживание к глобальному распределению) + сырые счётчики."""
        a = self.cfg['alpha_loc']; Ns = self.N.get(sloc, 0)
        if Ns:
            cnt = pd.Series(self.cl).map(self.pair[sloc]).fillna(0).values
            p = (cnt + a * self.cl_base) / (Ns + a)
        else:
            p = np.where(self.cl == sloc, 0.77, 0.23 * self.cl_base)    # локации нет в логе: 77% — та же локация (среднее по train)
        return np.log(p + 1e-7).astype(np.float32), (cnt if Ns else (self.cl == sloc).astype(float))

    def mc_scores(self, q, qts):
        """-> (exact P(mc|q) vector over corpus, token-NB log score vector over corpus, n_exact)"""
        nq = self.q_n.get(q, 0)
        ex = np.zeros(len(self.mc_list), np.float32)
        if nq:
            s = self.q_mc[q]; ex[[self.mc_pos[m] for m in s.index]] = s.values / nq
        ids = [self.tok_vocab[t] for t in set(qts) if t in self.tok_vocab]
        if ids:
            rows = self.tok_mc[ids].toarray(); nt = self.tok_n[ids][:, None]
            p = (rows + 2 * self.mc_prior) / (nt + 2)
            nb = np.log(p).mean(0) - np.log(self.mc_prior)          # средний log-lift по словам запроса
        else:
            nb = np.zeros(len(self.mc_list), np.float32)
        return ex[self.cmc], nb[self.cmc].astype(np.float32), nq

    def build(self, qdf, dense, with_labels=True, rel_col='rel', qplain=None):
        """Кандидаты + признаки для всех запросов qdf. dense: {'d1': (Q, D), 'd2': ...} — первый энкодер основной (он и для кандидатов).
        qplain — эмбеддинги запросов без фильтров (для kNN по логу). with_labels — добавить y (для обучения/валидации)."""
        cfg = self.cfg; out = []; t0 = time.time()
        qdf = qdf.reset_index(drop=True)
        cpos = {c: j for j, c in enumerate(self.corpus)}
        for qi, r in enumerate(qdf.itertuples()):
            q = r.search_query; qts = toks(q); s = r.search_location_id
            lp, lcnt = self.loc_logp(s)
            bm = {f: self.idx[f].score(qts) for f in self.idx}
            bmax = bm['all'].max() + 1e-6
            dn = {k: (D.astype(np.float32) @ Q[qi]) if D.dtype != np.float32 else D @ Q[qi] for k, (Q, D) in dense.items()}
            main = list(dense)[0] if dense else None
            # ---------------- 1. кандидаты: объединение нескольких ретриверов
            cand = set()
            def top(v, k):
                k = min(k, len(v) - 1); return np.argpartition(-v, k)[:k]
            cand.update(top(bm['all'] + cfg['wl_bm'] * bmax * lp, cfg['k_bm_loc']))
            cand.update(top(bm['title'] + cfg['wl_bm'] * (bm['title'].max() + 1e-6) * lp, cfg['k_title_loc']))
            cand.update(top(bm['all'], cfg['k_bm_glob']))
            if main:
                # косинусы разных энкодеров имеют разный масштаб — приводим к шкале e5-small, на которой подбирались веса
                d = dn[main]; d = d * (REF_STD / (d.std() + 1e-6))
                cand.update(top(d + 0.02 * lp, cfg['k_dn_loc']))
                cand.update(top(d, cfg['k_dn_glob']))
                cand.update(top(10 * d + bm['all'] / bmax + 0.2 * lp, cfg['k_hyb']))
            ch = np.asarray((self.CH @ self.chv.transform([norm(q)]).T).todense()).ravel()
            cand.update(top(ch + 0.02 * lp, cfg['k_chr']))
            ex, nb, nq = self.mc_scores(q, qts)
            mcs = np.maximum(nb, 0) + 3 * ex
            cand.update(top(mcs + 0.3 * lp + 0.05 * self.stat.i_reviews.values, cfg['k_mc']))   # популярные объявления вероятных микрокатегорий рядом
            if self.knn is not None and qplain is not None:
                kmc, kprf, kmax, kmean = self.knn_feats(qplain[qi], q, None)
                cand.update(top(kprf / (kprf.max() + 1e-6) + 2 * kmc + 0.3 * lp, cfg['k_mc']))
            c = np.fromiter(cand, dtype=np.int64)
            # ---------------- 2. признаки кандидатов
            f = {'qi': np.full(len(c), qi, np.int32), 'ci': c}
            for k, v in bm.items():     # текстовые: сырой BM25 и нормированный на максимум по корпусу
                f[f'bm_{k}'] = v[c]; f[f'bmn_{k}'] = v[c] / (v.max() + 1e-6)
            for k, v in dn.items():     # dense: косинус и отставание от 50-го по корпусу (устойчиво к калибровке)
                f[f'dn_{k}'] = v[c]; f[f'dnr_{k}'] = v[c] - np.partition(v, -50)[-50]
            # локация: приор, нормированный счётчик, «та же локация», расстояние до центра локации поиска (самый важный признак ранкера)
            f['loc_lp'] = lp[c]; f['loc_cnt'] = lcnt[c] * self.scale; f['same_loc'] = (self.cl[c] == s).astype(np.int8)
            cen = self.cent.loc[s] if s in self.cent.index else (self.cent_item.loc[s] if s in self.cent_item.index else None)
            if cen is not None:
                f['dist'] = np.sqrt((self.lat[c] - cen.lat) ** 2 + ((self.lon[c] - cen.lon) * np.cos(np.radians(cen.lat))) ** 2)
            else:
                f['dist'] = np.full(len(c), np.nan)
            f['chr'] = ch[c]; f['chr_n'] = ch[c] / (ch.max() + 1e-6)
            if self.knn is not None and qplain is not None:
                f['knn_mc'] = kmc[c]; f['knn_prf'] = kprf[c]; f['knn_prf_n'] = kprf[c] / (kprf.max() + 1e-6)
                f['knn_max'] = np.full(len(c), kmax, np.float32); f['knn_mean'] = np.full(len(c), kmean, np.float32)
            f['mc_ex'] = ex[c]; f['mc_nb'] = nb[c]
            qs = set(qts); nqt = max(len(qs), 1)
            f['cov_title'] = self.cov('title', qts, c) / nqt
            f['cov_all'] = self.cov('all', qts, c) / nqt
            # история объявления в логе: сколько раз выбирали, насколько прошлые запросы похожи на текущий, был ли ровно такой запрос
            f['hist_n'] = self.hist_n[c] * self.scale
            f['hist_cov'] = np.array([len(qs & self.hist_toks[j]) / nqt if j in self.hist_toks else -1 for j in c], np.float32)
            f['hist_exact'] = np.array([(q in self.hist_q[j]) if j in self.hist_q else 0 for j in c], np.int8)
            # filters
            # совпадение с фильтрами поиска: 1/0, либо -1, если такого фильтра в запросе нет
            sd = param_dict(r.search_infm_params_text)
            sv = (sd.get('Вид услуги') or [''])[0]; stp = (sd.get('Тип услуги') or [''])[0]
            f['f_vid'] = (self.i_vid[c] == sv).astype(np.int8) if sv else np.full(len(c), -1, np.int8)
            f['f_tip'] = np.array([stp in self.i_tip[j] for j in c], np.int8) if stp else np.full(len(c), -1, np.int8)
            f['f_rating'] = (np.nan_to_num(self.stat.i_rating.values[c]) >= 4).astype(np.int8) if 'Рейтинг пользователя' in r.search_infm_params_text else np.full(len(c), -1, np.int8)
            pairs = [(k, v) for k, vs in sd.items() for v in vs if k not in ('Вид услуги', 'Тип услуги', 'Рейтинг пользователя')]
            if pairs:
                f['f_other'] = np.array([np.mean([(k in self.i_pd[j]) and ((v in self.i_pd[j][k]) if v else True) for k, v in pairs]) for j in c], np.float32)
            else:
                f['f_other'] = np.full(len(c), -1, np.float32)
            df = pd.DataFrame(f)
            for k in self.stat: df[k] = self.stat[k].values[c]
            # query-level
            df['q_ntok'] = len(qts); df['q_nexact'] = nq * self.scale; df['q_hasf'] = int(bool(r.search_infm_params_text))
            df['q_cat0'] = int(r.search_category == 0); df['q_Nloc'] = self.N.get(s, 0) * self.scale
            df['q_bmax'] = bmax
            if with_labels:
                rs = {cpos[x] for x in getattr(r, rel_col)}
                df['y'] = df.ci.isin(rs).astype(np.int8)
            out.append(df)
            if qi % 500 == 0: print(qi, f'{time.time()-t0:.0f}s', flush=True)
        return pd.concat(out, ignore_index=True)


def add_ranks(df):
    """Within-query rank features (calibration-invariant).

    Ранги внутри запроса не зависят от масштаба скоров — это делает ранкер устойчивым к смене энкодера
    (модель на T-логе при обучении ранкера vs модель на всём train на бенчмарке)."""
    cols = [c for c in df.columns if c.startswith(('bm_', 'dn_')) or c in ('chr', 'mc_nb', 'loc_lp', 'cov_all')]
    g = df.groupby('qi')
    for c in cols:
        df[f'rk_{c}'] = g[c].rank(ascending=False, method='min').astype(np.float32)
    return df

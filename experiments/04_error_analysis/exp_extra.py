# A/B признаков «нелокальности»: -0.17 п.п. — не вошли.
# Запуск из корня репозитория: python experiments/04_error_analysis/exp_extra.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks; from extra_feats import Extra
NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
items = pd.read_parquet('data/items.parquet', columns=['iid','item_location_id','item_microcat_id','item_title_raw','item_description_raw','item_infm_params_text'])
corpus = np.load('data/val_corpus.npy'); T = pd.read_parquet('data/T_log.parquet')
val = pd.read_parquet('data/val_q.parquet'); rk = pd.read_parquet('data/rk_q.parquet').iloc[:15000]
E = Extra(corpus, T, items); del items
def load(name, qdf):
    F = pd.read_parquet(f'data/feats_v2_{name}.parquet')
    if name == 'rk': F = F[F.qi < NQ]
    for c in F.columns:
        if F[c].dtype == np.float64: F[c] = F[c].astype(np.float32)
    return E.add(add_ranks(F), qdf)
tr = load('rk', rk); va = load('val', val)
nrel = val.rel.str.len().values
tr = tr[tr.groupby('qi').y.transform('max') > 0].sort_values('qi')
def recall(scores):
    d = va[['qi', 'y']].copy(); d['s'] = scores; d['r'] = d.groupby('qi').s.rank(ascending=False, method='first')
    hit = d[(d.r <= 50) & (d.y == 1)].groupby('qi').size()
    return (hit.reindex(range(len(nrel))).fillna(0).values / nrel).mean()
P = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambdarank_truncation_level=60, verbose=-1, num_threads=6)
new = ['q_tfar_mean', 'q_tfar_max', 'q_far', 'q_remote', 'i_mcfar', 'i_remote', 'i_remote_t', 'i_city', 'i_zones']
base = [c for c in tr.columns if c not in ['qi', 'ci', 'y', 'i_cat114'] + new]
grp = tr.groupby('qi').size().values
for name, fs in [('base', base), ('+extra', base + new)]:
    rs = []
    for seed in [0, 1]:
        m = lgb.train(dict(P, seed=seed), lgb.Dataset(tr[fs], tr.y, group=grp), 300)
        rs.append(recall(m.predict(va[fs])))
    print(name, np.round(rs, 4), round(np.mean(rs), 4), flush=True)
imp = pd.Series(m.feature_importance('gain'), index=base + new); print((imp / imp.sum()).loc[new].round(4).to_dict())

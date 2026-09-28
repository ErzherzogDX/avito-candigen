# Финальный ранкер локальной версии: ансамбль 3 сидов на rk + val.
# Запуск из корня репозитория: python experiments/03_local_pipeline/train_final_ranker.py (нужны data/ из scripts/prep.py … split.py)
"""Final ranker: LightGBM lambdarank on rk + val pseudo-benchmark queries, seed ensemble.
usage: train_final_ranker.py FEATS_TAG [n_rounds] [seeds]"""
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 400; seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
parts = []
for name in ['rk', 'val']:
    F = pd.read_parquet(f'data/feats_{tag}_{name}.parquet')
    F = F[F.groupby('qi').y.transform('max') > 0]
    F['qi'] = F.qi + (0 if name == 'rk' else 10**6)
    parts.append(add_ranks(F))
tr = pd.concat(parts, ignore_index=True).sort_values('qi')
feats = [c for c in tr.columns if c not in ('qi', 'ci', 'y', 'i_cat114')]
grp = tr.groupby('qi').size().values
P = dict(objective='lambdarank', learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambdarank_truncation_level=60, verbose=-1, num_threads=8)
paths = []
for sd in range(seeds):
    m = lgb.train(dict(P, seed=sd), lgb.Dataset(tr[feats], tr.y, group=grp), n)
    p = f'models/lgb_final_{tag}_s{sd}.txt'; m.save_model(p); paths.append(p)
print(','.join(paths))

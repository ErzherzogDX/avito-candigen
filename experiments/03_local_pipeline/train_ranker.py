# LightGBM LambdaRank на признаках: 0.925 (zero-shot) -> 0.944 (дообученный e5-small, char-3gram, kNN, микрокатегории).
# Запуск из корня репозитория: python experiments/03_local_pipeline/train_ranker.py (нужны data/ из scripts/prep.py … split.py)
import sys, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, 'src'); from pipeline import add_ranks
tag = sys.argv[1]; obj = sys.argv[2] if len(sys.argv) > 2 else 'lambdarank'
drop = set(sys.argv[3].split(',')) if len(sys.argv) > 3 and sys.argv[3] else set()
tr = pd.read_parquet(f'data/feats_{tag}_rk.parquet'); va = pd.read_parquet(f'data/feats_{tag}_val.parquet')
tr = add_ranks(tr); va = add_ranks(va)
val = pd.read_parquet('data/val_q.parquet'); nrel = val.rel.str.len().values
feats = [c for c in tr.columns if c not in ('qi', 'ci', 'y') and c not in drop]
pos_q = tr.groupby('qi').y.transform('max') > 0
trp = tr[pos_q].sort_values('qi')
grp = trp.groupby('qi').size().values
params = dict(objective=obj, learning_rate=0.05, num_leaves=63, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambdarank_truncation_level=60, verbose=-1, num_threads=8)
if obj == 'lambdarank': params['eval_at'] = [50]
dtr = lgb.Dataset(trp[feats], trp.y, group=grp if obj == 'lambdarank' else None)
def recall(scores, df):
    d = df[['qi', 'y']].copy(); d['s'] = scores
    d['r'] = d.groupby('qi').s.rank(ascending=False, method='first')
    hit = d[(d.r <= 50) & (d.y == 1)].groupby('qi').size()
    return (hit.reindex(range(len(nrel))).fillna(0).values / nrel).mean()
print('val union recall', recall(np.zeros(len(va)) + va.y.values, va), 'n feats', len(feats))
for n in [200, 400, 800]:
    m = lgb.train(params, dtr, num_boost_round=n)
    print(obj, n, 'val R@50', round(recall(m.predict(va[feats]), va), 4), flush=True)
imp = pd.Series(m.feature_importance('gain'), index=feats).sort_values(ascending=False)
print((imp / imp.sum()).round(4).head(25).to_string())
m.save_model(f'models/lgb_{tag}_{obj}.txt')

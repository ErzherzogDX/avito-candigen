"""Strict format validation of answer.csv against the task rules.

Проверяет всё, что требует условие: заголовок query_id,answer; ровно по одной строке на каждый query_id бенчмарка;
не больше 50 item_id в строке, без повторов; каждый item_id — 16 символов 0-9a-f и существует в benchmark_items.parquet.
Использование:  python scripts/check_answer.py answer.csv
"""
import sys, re, pandas as pd
p = sys.argv[1]
raw = open(p, encoding='utf-8').read().splitlines()
assert raw[0] == 'query_id,answer', f'bad header {raw[0]!r}'
a = pd.read_csv(p, dtype=str, keep_default_na=False)          # dtype=str: не терять ведущие нули и не уходить в экспоненту
bq = pd.read_parquet('benchmark_queries.parquet', columns=['query_id'])
corpus = set(pd.read_parquet('benchmark_items.parquet', columns=['item_id']).item_id)
assert list(a.columns) == ['query_id', 'answer']
assert len(a) == len(bq) and a.query_id.is_unique and set(a.query_id) == set(bq.query_id), 'query_id set mismatch'
assert a.query_id.str.len().eq(16).all()
hexre = re.compile(r'^[0-9a-f]{16}$')
n = []
for s in a.answer:
    ids = s.split(' ') if s else []
    assert len(ids) <= 50 and len(ids) == len(set(ids)), 'dup or >50'
    assert all(hexre.match(i) and i in corpus for i in ids), 'bad item id'
    n.append(len(ids))
print('OK', len(a), 'rows; ids/row min', min(n), 'mean', sum(n) / len(n))

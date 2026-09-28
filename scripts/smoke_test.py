"""Smoke-тест: весь solution.ipynb на крошечных подвыборках (SMOKE = True) — проверка кода за несколько минут.
Запуск из корня репозитория:  python scripts/smoke_test.py   (аргумент — имя jupyter-kernel, по умолчанию python3)"""
import nbformat, sys, time
from nbclient import NotebookClient
nb = nbformat.read('solution.ipynb', as_version=4)
src = nb.cells[1].source
assert 'SMOKE = False' in src
nb.cells[1].source = src.replace('SMOKE = False', 'SMOKE = True')
t0 = time.time()
client = NotebookClient(nb, timeout=7200, kernel_name=sys.argv[1] if len(sys.argv) > 1 else 'python3', resources={'metadata': {'path': '.'}})
try:
    client.execute()
    print('SMOKE OK', round(time.time() - t0), 's')
finally:
    nbformat.write(nb, 'results/smoke_executed.ipynb')

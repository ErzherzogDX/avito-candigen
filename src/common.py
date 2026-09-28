"""Общие текстовые утилиты: нормализация и стемминг, разбор параметров объявления, разреженный BM25, метрика.

Используется и при подготовке данных (build_tok.py, item_text.py), и при сборке кандидатов/признаков (pipeline.py).
"""
import re, numpy as np, pandas as pd, scipy.sparse as sp
import Stemmer

# ---------------------------------------------------------------------------------------------
# Токенизация. Запросы короткие и морфологически разнообразные («баня на дровах» / «бани дровяные»),
# поэтому BM25 работает по основам слов: Snowball-стеммер для русского и английского (PyStemmer, C-реализация — быстро).
# ---------------------------------------------------------------------------------------------
_ru = Stemmer.Stemmer('russian'); _en = Stemmer.Stemmer('english')
TOK = re.compile(r'[a-zа-я0-9]+')          # слова из латиницы, кириллицы и цифр (после приведения к нижнему регистру)
_cache = {}                                # кеш «слово -> основа»: словарь маленький, а слов в корпусе сотни миллионов
def stem(t):
    s = _cache.get(t)
    if s is None:
        s = _en.stemWord(t) if t.isascii() else _ru.stemWord(t)   # латиница (бренды, модели техники) — английский стеммер
        _cache[t] = s
    return s
def norm(s): return s.lower().replace('ё', 'е')          # «ё» и «е» в запросах пишут вперемешку
def toks(s): return [stem(t) for t in TOK.findall(norm(s))]   # текст -> список основ (для BM25 и приоров)
def toks_raw(s): return TOK.findall(norm(s))                  # текст -> список слов без стемминга

# ---------------------------------------------------------------------------------------------
# Параметры объявления/фильтры поиска приходят одной строкой вида «ключ значение ключ значение …»,
# например: «Вид услуги Красота, здоровье Тип услуги Маникюр, педикюр Место оказания услуг Москва, …».
# Разбираем её по известному списку ключей (длинные ключи проверяются раньше коротких: «Тип услуги автосервиса» до «Тип услуги»).
# Последние пять ключей появились только в периоде бенчмарка (новые фильтры площадки) — добавлены после анализа ошибок.
# ---------------------------------------------------------------------------------------------
PARAM_KEYS = ['Место оказания услуг','Тип услуги автосервиса','Тип услуги','Тип стоимости за услугу','Тип стоимости','Гарантия','Опыт работы',
 'Где вы оказываете услуги','Название услуги','Специальность или сфера','Время для связи, дни недели','Время для связи от','Время для связи до',
 'График работы от','График работы до','Время работы, с','Время работы, до','Онлайн-запись','Ваши клиенты','Кто оказывает услуги','Дополнительно',
 'Начальная цена','Стоимость','Вид услуги','Продолжительность','Куда выезжаете','Районы','Работа по договору','Признак предзаполнения прайс листа',
 'Инструменты','Минимальное время заказа','Рейтинг пользователя',
 'Предмет или специальность','Срочная услуга (мультистатус)','Поиск по слотам','Преподаватель','Рабочие дни']
_kre = re.compile(r'(?:^| )(' + '|'.join(sorted(map(re.escape, PARAM_KEYS), key=len, reverse=True)) + r')(?= |$)')
def parse_params(s):
    """-> list of (key, value) in order."""
    out = []; ms = list(_kre.finditer(s))
    for i, m in enumerate(ms):
        end = ms[i+1].start() if i+1 < len(ms) else len(s)      # значение — текст до следующего ключа
        out.append((m.group(1), s[m.end():end].strip()))
    return out
# ключи, описывающие саму услугу (попадают в текст для символьных n-грамм)
SERVICE_KEYS = {'Вид услуги','Тип услуги','Тип услуги автосервиса','Название услуги','Специальность или сфера','Услуга','Дополнительно'}
def param_service_text(s):
    return ' '.join(v for k, v in parse_params(s) if k in SERVICE_KEYS and v)
def param_dict(s):
    """ключ -> список значений (ключ может повторяться: несколько «Где вы оказываете услуги»)."""
    d = {}
    for k, v in parse_params(s): d.setdefault(k, []).append(v)
    return d

class BM25:
    """Sparse BM25 over a list of token lists. score(qtoks) -> dense vector over docs.

    Матрица весов W (документы x термины) считается один раз; скор запроса = сумма столбцов W по термам запроса.
    Одна операция со срезом разреженной матрицы — миллисекунды на корпус в ~190 тыс. объявлений."""
    def __init__(self, docs, k1=1.2, b=0.75, vocab=None):
        if vocab is None:
            vocab = {}
            for d in docs:
                for t in d:
                    if t not in vocab: vocab[t] = len(vocab)
        self.vocab = vocab
        rows, cols = [], []
        for i, d in enumerate(docs):
            for t in d:
                j = vocab.get(t)
                if j is not None: rows.append(i); cols.append(j)
        n = len(docs)
        tf = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(vocab)))
        tf.sum_duplicates()                                       # частоты терминов в документе
        dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean() if n else 1   # длины документов
        df = np.bincount(tf.indices, minlength=len(vocab))        # документная частота
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)     # стандартная нормировка BM25 по длине документа
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.W = sp.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)   # CSC: быстрый срез по термам
        self.n = n
    def qvec(self, qtoks):
        ids = [self.vocab[t] for t in set(qtoks) if t in self.vocab]
        return ids
    def score(self, qtoks):
        ids = self.qvec(qtoks)
        if not ids: return np.zeros(self.n, np.float32)
        return np.asarray(self.W[:, ids].sum(1)).ravel()

def recall_at(preds, rels, k=50):
    """preds: list of arrays of iids (ranked); rels: list of lists.  Средняя по запросам доля найденных релевантных — Recall@k."""
    r = []
    for p, rel in zip(preds, rels):
        s = set(p[:k]); r.append(sum(1 for x in rel if x in s) / len(rel))
    return float(np.mean(r))

# Ключи-«шум» для текстового поиска: адрес, расписание, цены и т. п. — выбрасываются из текста параметров
# (адрес при этом учитывается отдельно, через локацию и координаты объявления).
DROP_KEYS = {'Место оказания услуг','Время для связи, дни недели','Время для связи от','Время для связи до','График работы от','График работы до',
             'Время работы, с','Время работы, до','Стоимость','Опыт работы','Районы','Признак предзаполнения прайс листа','Минимальное время заказа',
             'Тип стоимости за услугу','Тип стоимости','Начальная цена','Продолжительность'}
def param_text_clean(s):
    return ' '.join(v for k, v in parse_params(s) if k not in DROP_KEYS and v)

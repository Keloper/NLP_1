# удаление пунктуации, лишних символом
# приведение слов к нижнему регистру

import re 
from gensim.models import FastText

def clean_wap_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Приводим к нижнему регистру
    text = text.lower()

    # 2. Заменяем 'ё' на 'е' (стандарт де-факто для FastText в русском языке)
    text = text.replace("ё", "е")

    # 3. Заменяем переносы строк, табуляции и длинные/двойные тире на ПРОБЕЛЫ,
    # чтобы слова не склеились
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"[-—–]{2,}", " ", text)  # ловит --, ---, длинные тире

    # 4. Удаляем всю пунктуацию, спецсимволы и цифры.
    # Оставляем только русские и английские буквы и пробелы.
    # (Все нежелательные символы заменяем на пробел)
    text = re.sub(r"[^a-zа-я\s]", " ", text)

    # 5. Схлопываем цепочки пробелов в один и обрезаем края
    text = re.sub(r"\s+", " ", text).strip()

    return text

def build_fasttext_model(vector_size=100, window=5, min_count=1):
  return FastText(vector_size=vector_size, window=window, min_count=min_count) 


def train_fasttext_model():
  return model.train(corpus_iterable=common_texts, total_examples=len(common_texts), epochs=10)  # train

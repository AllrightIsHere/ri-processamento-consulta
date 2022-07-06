from textwrap import indent
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import string
from nltk.tokenize import word_tokenize
import os
import json

from regex import F


class Cleaner:
    def __init__(self, stop_words_file: str, language: str,
                 perform_stop_words_removal: bool, perform_accents_removal: bool,
                 perform_stemming: bool):
        self.set_stop_words = self.read_stop_words(stop_words_file)

        self.stemmer = SnowballStemmer(language)
        in_table = "áéíóúâêôçãẽõü"
        out_table = "aeiouaeocaeou"
        # altere a linha abaixo para remoção de acentos (Atividade 11)
        self.accents_translation_table = str().maketrans(in_table, out_table)
        self.set_punctuation = set(string.punctuation)

        # flags
        self.perform_stop_words_removal = perform_stop_words_removal
        self.perform_accents_removal = perform_accents_removal
        self.perform_stemming = perform_stemming

    def html_to_plain_text(self, html_doc: str) -> str:
        return BeautifulSoup(html_doc, features="lxml").get_text()

    @staticmethod
    def read_stop_words(str_file) -> set:
        set_stop_words = set()
        with open(str_file, encoding='utf-8') as stop_words_file:
            for line in stop_words_file:
                arr_words = line.split(",")
                [set_stop_words.add(word) for word in arr_words]
        return set_stop_words

    def is_stop_word(self, term: str):
        return term in self.set_stop_words

    def word_stem(self, term: str):
        return self.stemmer.stem(term)

    def remove_accents(self, term: str) -> str:
        return term.translate(self.accents_translation_table)

    def preprocess_word(self, term: str) -> str or None:
        if term in self.set_punctuation:
            return None

        if self.perform_stop_words_removal and self.is_stop_word(term):
            return None

        return self.word_stem(term) if self.perform_stemming else term

    def preprocess_text(self, text: str) -> str or None:
        return self.remove_accents(text.lower()) if self.perform_accents_removal else text.lower()


class HTMLIndexer:
    cleaner = Cleaner(stop_words_file="stopwords.txt",
                      language="portuguese",
                      perform_stop_words_removal=True,
                      perform_accents_removal=True,
                      perform_stemming=True)

    def __init__(self, index):
        self.index = index
        self.debug_set = []

    def text_word_count(self, plain_text: str):
        dic_word_count = {}

        plain_text = self.cleaner.preprocess_text(plain_text)

        tokens = word_tokenize(plain_text, language="portuguese")

        for token in tokens:
            term = self.cleaner.preprocess_word(token)

            if term is not None:
                if term not in dic_word_count:
                    dic_word_count[term] = 0
                dic_word_count[term] += 1

        return dic_word_count

    def index_text(self, doc_id: int, text_html: str):
        terms_count = self.text_word_count(
            self.cleaner.html_to_plain_text(text_html))

        if "horizonte" in terms_count:
            self.debug_set.append([doc_id, terms_count["horizonte"]])
        # print(terms_count)
        for term, count in terms_count.items():
            self.index.index(term, doc_id, count)

    def index_text_dir(self, path: str):
        # print(self.cleaner.set_stop_words)
        for str_sub_dir in os.listdir(path):
            path_sub_dir = f"{path}/{str_sub_dir}"

            for name_file in os.listdir(path_sub_dir):
                path_file = f'{path_sub_dir}/{name_file}'
                # print(path_file)
                with open(path_file, 'r') as arquivo:
                    html_text = arquivo.read()
                    # print(html_text)
                    name = name_file.replace('.html', '')
                    # print(html_text)
                    self.index_text(int(name), html_text)

                    arquivo.close()

        self.index.finish_indexing()

        file_debug = open("debug_nosso.json", "w")

        json.dump(self.debug_set, file_debug, indent=4)

        file_debug.close()

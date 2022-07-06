from curses import termattrs
from IPython.display import clear_output
from typing import List, Set, Union
from abc import abstractmethod
from functools import total_ordering
from os import path
import os
import pickle
import gc


class Index:
    def __init__(self):
        self.dic_index = {}
        self.set_documents = set()

    def index(self, term: str, doc_id: int, term_freq: int):
        if term not in self.dic_index:
            int_term_id = len(self.dic_index)
            self.dic_index[term] = self.create_index_entry(int_term_id)
        else:
            int_term_id = self.get_term_id(term)

        self.set_documents.add(doc_id)

        self.add_index_occur(
            self.dic_index[term], doc_id, int_term_id, term_freq)

    @property
    def vocabulary(self) -> List[str]:
        return self.dic_index.keys()

    @property
    def document_count(self) -> int:
        return len(self.set_documents)

    @abstractmethod
    def get_term_id(self, term: str):
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def create_index_entry(self, termo_id: int):
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def add_index_occur(self, entry_dic_index, doc_id: int, term_id: int, freq_termo: int):
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def get_occurrence_list(self, term: str) -> List:
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    @abstractmethod
    def document_count_with_term(self, term: str) -> int:
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def finish_indexing(self):
        pass

    def write(self, arq_index: str):
        #print(f'{arq_index}: {self.dic_index}')
        arquivo = open(arq_index, 'wb')
        pickle.dump(self, arquivo)
        arquivo.close()

    @staticmethod
    def read(arq_index: str):
        arquivo = open(arq_index, 'rb')
        obj = pickle.load(arquivo)
        arquivo.close()

        #print(f'read: {dic_index}')

        return obj

    def __str__(self):
        arr_index = []
        for str_term in self.vocabulary:
            arr_index.append(
                f"{str_term} -> {self.get_occurrence_list(str_term)}")

        return "\n".join(arr_index)

    def __repr__(self):
        return str(self)


@total_ordering
class TermOccurrence:
    def __init__(self, doc_id: int, term_id: int, term_freq: int):
        self.doc_id = doc_id
        self.term_id = term_id
        self.term_freq = term_freq

    def write(self, idx_file):
        idx_file.write(self.doc_id.to_bytes(4, byteorder="big"))
        idx_file.write(self.term_id.to_bytes(4, byteorder="big"))
        idx_file.write(self.term_freq.to_bytes(4, byteorder="big"))

    def __hash__(self):
        return hash((self.doc_id, self.term_id))

    def __eq__(self, other_occurrence: "TermOccurrence"):
        if not other_occurrence:
            return False

        return self.term_id == other_occurrence.term_id and self.doc_id == other_occurrence.doc_id

    def __lt__(self, other_occurrence: "TermOccurrence"):
        if other_occurrence is None:
            return True
            # raise TypeError("Comparison not suported for NoneType.")

        if self.term_id != other_occurrence.term_id:
            return self.term_id < other_occurrence.term_id
        else:
            return self.doc_id < other_occurrence.doc_id

    def __str__(self):
        return f"( doc: {self.doc_id} term_id:{self.term_id} freq: {self.term_freq})"

    def __repr__(self):
        return str(self)


# HashIndex é subclasse de Index
class HashIndex(Index):
    def get_term_id(self, term: str):
        return self.dic_index[term][0].term_id

    def create_index_entry(self, termo_id: int) -> List:
        return []

    def add_index_occur(self, entry_dic_index: List[TermOccurrence], doc_id: int, term_id: int, term_freq: int):
        entry_dic_index.append(TermOccurrence(
            doc_id=doc_id, term_freq=term_freq, term_id=term_id))

    def get_occurrence_list(self, term: str) -> List:
        return self.dic_index[term] if term in self.dic_index else []

    def document_count_with_term(self, term: str) -> int:
        return len(self.get_occurrence_list(term))


class TermFilePosition:
    def __init__(self, term_id: int, term_file_start_pos: int = None, doc_count_with_term: int = None):
        self.term_id = term_id

        # a serem definidos após a indexação
        self.term_file_start_pos = term_file_start_pos
        self.doc_count_with_term = doc_count_with_term

    def __str__(self):
        return f"term_id: {self.term_id}, doc_count_with_term: {self.doc_count_with_term}, term_file_start_pos: {self.term_file_start_pos}"

    def __repr__(self):
        return str(self)


class FileIndex(Index):
    TMP_OCCURRENCES_LIMIT = 1000000

    def __init__(self):
        super().__init__()

        self.lst_occurrences_tmp = [None]*FileIndex.TMP_OCCURRENCES_LIMIT
        self.idx_file_counter = 0
        self.str_idx_file_name = "occur_idx_file"

        # metodos auxiliares para verifica o tamanho da lst_occurrences_tmp
        self.idx_tmp_occur_last_element = -1
        self.idx_tmp_occur_first_element = 0

    def get_term_id(self, term: str):
        return self.dic_index[term].term_id

    def create_index_entry(self, term_id: int) -> TermFilePosition:
        return TermFilePosition(term_id)

    def add_index_occur(self, entry_dic_index: TermFilePosition, doc_id: int, term_id: int, term_freq: int):
        # complete aqui adicionando um novo TermOccurrence na lista lst_occurrences_tmp
        # não esqueça de atualizar a(s) variável(is) auxiliares apropriadamente
        self.idx_tmp_occur_last_element += 1
        self.lst_occurrences_tmp[self.idx_tmp_occur_last_element] = TermOccurrence(
            doc_id, term_id, term_freq)

        if self.get_tmp_occur_size() >= self.TMP_OCCURRENCES_LIMIT:
            self.save_tmp_occurrences()

    def get_tmp_occur_size(self) -> int:
        return self.idx_tmp_occur_last_element - self.idx_tmp_occur_first_element + 1 if self.idx_tmp_occur_last_element > -1 else 0

    def next_from_list(self) -> TermOccurrence:
        if self.get_tmp_occur_size() > 0:
            # obtenha o proximo da lista e armazene em nex_occur
            # não esqueça de atualizar a(s) variável(is) auxiliares apropriadamente
            next_occur = self.lst_occurrences_tmp[self.idx_tmp_occur_first_element]
            self.lst_occurrences_tmp[self.idx_tmp_occur_first_element] = None
            self.idx_tmp_occur_first_element += 1
            return next_occur
        else:
            return None

    def next_from_file(self, file_pointer) -> TermOccurrence:
        # next_from_file = pickle.load(file_idx)
        if file_pointer is None:
            return None

        bytes_doc_id = file_pointer.read(4)
        if not bytes_doc_id:
            return None
        doc_id = int.from_bytes(bytes_doc_id, byteorder='big')

        bytes_term_id = file_pointer.read(4)
        if not bytes_term_id:
            return None
        term_id = int.from_bytes(bytes_term_id, byteorder='big')

        bytes_term_freq = file_pointer.read(4)
        if not bytes_term_freq:
            return None
        term_freq = int.from_bytes(bytes_term_freq, byteorder='big')

        return TermOccurrence(doc_id=doc_id, term_id=term_id, term_freq=term_freq)

    def save_tmp_occurrences(self):

        # Ordena pelo term_id, doc_id
        #    Para eficiência, todo o código deve ser feito com o garbage collector desabilitado gc.disable()
        gc.disable()

        occurrences = self.lst_occurrences_tmp[self.idx_tmp_occur_first_element:self.idx_tmp_occur_last_element+1]
        occurrences.sort()

        self.lst_occurrences_tmp[self.idx_tmp_occur_first_element:
                                 self.idx_tmp_occur_last_element+1] = occurrences

        # print(self.lst_occurrences_tmp, self.idx_tmp_occur_first_element,
        #      self.idx_tmp_occur_last_element)

        """comparar sempre a primeira posição
        da lista com a primeira posição do arquivo usando os métodos next_from_list e next_from_file
        e use o método write do TermOccurrence para armazenar cada ocorrencia do novo índice ordenado"""

        file_idx = None if self.idx_file_counter == 0 else open(
            self.str_idx_file_name, "rb")
        # print(self.idx_file_counter)
        self.idx_file_counter += 1
        # print(self.idx_file_counter)

        self.str_idx_file_name = f'occur_idx_file_{self.idx_file_counter}.idx'

        new_file_idx = open(self.str_idx_file_name, "wb")

        next_list = self.next_from_list()
        next_file = self.next_from_file(file_idx)

        # print(file_idx)

        while next_file and next_list:
            if next_list > next_file:
                next_file.write(new_file_idx)
                next_file = self.next_from_file(file_idx)
            else:
                next_list.write(new_file_idx)
                next_list = self.next_from_list()

        while next_list:
            next_list.write(new_file_idx)
            next_list = self.next_from_list()

        while next_file:
            next_file.write(new_file_idx)
            next_file = self.next_from_file(file_idx)

        self.idx_tmp_occur_first_element = 0
        self.idx_tmp_occur_last_element = -1
        new_file_idx.close()
        if file_idx is not None:
            file_idx.close()

        gc.enable()

    def finish_indexing(self):
        if self.get_tmp_occur_size() > 0:
            self.save_tmp_occurrences()

        # Sugestão: faça a navegação e obetenha um mapeamento
        # id_termo -> obj_termo armazene-o em dic_ids_por_termo
        # obj_termo é a instancia TermFilePosition correspondente ao id_termo
        dic_ids_por_termo = {}

        # print(self.dic_index.items())

        for str_term, obj_term in self.dic_index.items():
            # print(str_term)
            dic_ids_por_termo[obj_term.term_id] = obj_term

        # print(dic_ids_por_termo.items())

        count_term = 0

        with open(self.str_idx_file_name, 'rb') as idx_file:
            # navega nas ocorrencias para atualizar cada termo em dic_ids_por_termo
            # apropriadamente
            term_atual = self.next_from_file(idx_file)
            term_anterior = term_atual

            if term_atual.term_id not in dic_ids_por_termo.keys():
                dic_ids_por_termo[term_atual.term_id] = self.create_index_entry(
                    term_atual.term_id)

            dic_ids_por_termo[term_atual.term_id].term_file_start_pos = 0

            while term_atual is not None:
                position = idx_file.tell()

                if term_atual.term_id == term_anterior.term_id:
                    count_term += 1
                else:
                    dic_ids_por_termo[term_anterior.term_id].doc_count_with_term = count_term

                    if term_atual.term_id not in dic_ids_por_termo.keys():
                        dic_ids_por_termo[term_atual.term_id] = self.create_index_entry(
                            term_atual.term_id)
                    dic_ids_por_termo[term_atual.term_id].term_file_start_pos = position - 12
                    count_term = 1

                # print(term_atual)
                term_anterior = term_atual
                term_atual = self.next_from_file(idx_file)

        idx_file.close()
        dic_ids_por_termo[term_anterior.term_id].doc_count_with_term = count_term
        # print(self.dic_index)

    def get_occurrence_list(self, term: str) -> List:
        term_pos = self.dic_index[term] if term in self.dic_index.keys() else [
        ]

        if term_pos == []:
            return []

        terms_occ = []

        arquivo = open(self.str_idx_file_name, 'rb')

        arquivo.seek(term_pos.term_file_start_pos)

        for i in range(term_pos.doc_count_with_term):
            terms_occ.append(self.next_from_file(arquivo))

        arquivo.close()

        return terms_occ

    def document_count_with_term(self, term: str) -> int:
        return self.dic_index[term].doc_count_with_term if term in self.dic_index else 0

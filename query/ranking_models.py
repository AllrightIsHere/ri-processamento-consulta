from typing import List
from abc import abstractmethod
from typing import List, Set, Mapping
from index.structure import TermOccurrence
import math
from enum import IntEnum
from tqdm import tqdm
import matplotlib.pyplot as plt


class IndexPreComputedVals():
    def __init__(self, index, skip_precompute=False):
        self.index = index
        if not skip_precompute:
            self.precompute_vals()

    @staticmethod
    def plot_terms_freq():
        x = [i for i in range(len(IndexPreComputedVals.term_freq_list))]
        y = [freq for freq, _ in IndexPreComputedVals.term_freq_list]

        _, ax = plt.subplots()

        ax.scatter(x, y)
        ax.set_yscale('log')

        plt.show()

    @staticmethod
    def print_top_idf():
        top10_melhores_idf = VectorRankingModel.idf_list[-10:]
        top10_melhores_idf.reverse()
        top10_piores_idf = VectorRankingModel.idf_list[:10]

        print("Top 10 melhores idf:")
        for i, item in enumerate(top10_melhores_idf):
            print(f"{i+1}: {item}")

        print("Top 10 piores idf:")
        for i, item in enumerate(top10_piores_idf):
            print(f"{i+1}: {item}")

    def calcule_tops_idf(self):
        vocab = self.index.vocabulary
        doc_count = len(self.index.set_documents)

        term_freq_list = []

        for term in tqdm(vocab):
            occur_list = self.index.get_occurrence_list(term)
            num_docs_with_term = len(occur_list)
            VectorRankingModel.idf_list.append(
                (VectorRankingModel.idf(doc_count, num_docs_with_term), term))
            term_occur_count = 0

            for occur in occur_list:
                term_occur_count += occur.term_freq

            term_freq_list.append((term_occur_count, term))

        VectorRankingModel.idf_list.sort()
        term_freq_list.sort(reverse=True)
        IndexPreComputedVals.term_freq_list = term_freq_list

    def precompute_vals(self):
        """
        Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """
        self.document_norm = {}

        vocab = self.index.vocabulary
        doc_count = len(self.index.set_documents)

        for term in tqdm(vocab):
            occur_list = self.index.get_occurrence_list(term)
            num_docs_with_term = len(occur_list)
            for occur in occur_list:
                doc_id = str(occur.doc_id)
                if doc_id not in self.document_norm.keys():
                    self.document_norm[doc_id] = 0
                self.document_norm[doc_id] += pow(VectorRankingModel.tf_idf(
                    doc_count, occur.term_freq, num_docs_with_term), 2)

        for doc in tqdm(self.document_norm.keys()):
            self.document_norm[doc] = math.sqrt(self.document_norm[doc])

        self.doc_count = self.index.document_count


class RankingModel():
    @abstractmethod
    def get_ordered_docs(self, query: Mapping[str, TermOccurrence],
                         docs_occur_per_term: Mapping[str, List[TermOccurrence]]) -> (List[int], Mapping[int, float]):
        raise NotImplementedError(
            "Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def rank_document_ids(self, documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key=lambda x: -documents_weight[x])
        return doc_ids


class OPERATOR(IntEnum):
    AND = 1
    OR = 2

# Atividade 1


class BooleanRankingModel(RankingModel):
    def __init__(self, operator: OPERATOR):
        self.operator = operator

    def intersection_all(self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]) -> List[int]:
        set_ids = set()

        for _, lst_occurrences in map_lst_occurrences.items():
            doc_list = map(lambda occur: occur.doc_id, lst_occurrences)
            if len(set_ids) > 0:
                set_ids = set_ids.intersection(doc_list)
            else:
                set_ids.update(doc_list)
        return set_ids

    def union_all(self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]) -> List[int]:
        set_ids = set()

        for _, lst_occurrences in map_lst_occurrences.items():
            set_ids.update(map(lambda occur: occur.doc_id, lst_occurrences))

        return set_ids

    def get_ordered_docs(self, query: Mapping[str, TermOccurrence],
                         docs_occur_per_term: Mapping[str, List[TermOccurrence]]) -> (List[int], Mapping[int, float]):
        """Considere que docs_occur_per_term possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(docs_occur_per_term), None
        else:
            return self.union_all(docs_occur_per_term), None

# Atividade 2


class VectorRankingModel(RankingModel):
    idf_list = []

    def __init__(self, idx_pre_comp_vals: IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term: int) -> float:
        if freq_term < 1:
            return 0

        return 1 + math.log(freq_term, 2)

    @staticmethod
    def idf(doc_count: int, num_docs_with_term: int) -> float:
        return math.log(doc_count / num_docs_with_term, 2)

    @staticmethod
    def tf_idf(doc_count: int, freq_term: int, num_docs_with_term) -> float:
        tf = VectorRankingModel.tf(freq_term)
        idf = VectorRankingModel.idf(doc_count, num_docs_with_term)
        # print(f"TF:{tf} IDF:{idf} n_i: {num_docs_with_term} N: {doc_count}")
        return tf*idf

    def get_ordered_docs(self, query: Mapping[str, TermOccurrence],
                         docs_occur_per_term: Mapping[str, List[TermOccurrence]]) -> (List[int], Mapping[int, float]):
        documents_norm = self.idx_pre_comp_vals.document_norm
        documents_weight = {}

        for term, query_occur in query.items():
            if term not in docs_occur_per_term.keys():
                continue
            docs_occur = docs_occur_per_term[term]
            num_docs_with_term = len(docs_occur)
            wiq = VectorRankingModel.tf_idf(
                self.idx_pre_comp_vals.doc_count, query_occur.term_freq, num_docs_with_term)
            for occur in docs_occur:
                doc_id = occur.doc_id
                if doc_id not in documents_weight.keys():
                    documents_weight[doc_id] = 0
                documents_weight[doc_id] += wiq * VectorRankingModel.tf_idf(
                    self.idx_pre_comp_vals.doc_count, occur.term_freq, num_docs_with_term)

        # print(documents_norm)
        for doc in documents_weight.keys():
            documents_weight[doc] = documents_weight[doc] / \
                documents_norm[str(doc)]

        # retona a lista de doc ids ordenados de acordo com o TF IDF
        return self.rank_document_ids(documents_weight), documents_weight

from index.structure import FileIndex, TermOccurrence
from query.processing import QueryRunner, VectorRankingModel, IndexPreComputedVals
from index.indexer import Cleaner
from typing import Mapping
import unittest


class ProcessingTest(unittest.TestCase):
    def setUp(self):
        self.index = FileIndex()
        self.index.index("adoro", 1, 1)
        self.index.index("vocês", 2, 3)
        self.index.index("espero", 2, 1)
        self.index.index("que", 3, 1)
        self.index.index("vocês", 3, 1)
        self.index.index("estejam", 3, 1)
        self.index.index("se", 3, 1)
        self.index.index("divertindo", 1, 4)

        self.index.finish_indexing()
        cleaner = Cleaner(stop_words_file="stopwords.txt", language="portuguese",
                          perform_stop_words_removal=False, perform_accents_removal=False,
                          perform_stemming=False)
        precomp = IndexPreComputedVals(self.index)
        self.queryRunner = QueryRunner(
            VectorRankingModel(precomp), self.index, cleaner)

    def test_count_top_n_relevant(self):
        arr_lists = [[1, 2, 3, 4, 5, 30, 23, 234, 32, 32, 3, 2, 10, 20],
                     [-1, -2, -2],
                     [14, 23, 21, 1, 10, 20, 4],
                     []
                     ]
        set_relevantes = {1, 4, 5, 10, 20}
        resp_esperada_por_top_n = {1: [1, 0, 0, 0],
                                   3: [1, 0, 0, 0],
                                   5: [3, 0, 2, 0],
                                   14: [5, 0, 4, 0]}

        for n, arr_resp_esperada in resp_esperada_por_top_n.items():
            #print(f"TOP {n}")
            for i, resp_esperada in enumerate(arr_resp_esperada):
                resposta = self.queryRunner.count_topn_relevant(
                    n, arr_lists[i], set_relevantes)
                self.assertEqual(
                    resp_esperada, resposta, msg=f"# de relevantes esperadas top {n}: {resp_esperada} resposta obtida: {resposta}")

    def test_precision_recall(self):
        lst_docs = [1, 2, 3, 4, 5, 6, 7, 9, 11]
        relevant_docs = {1, 3, 5, 7}
        precisao_esperada = 0.66667
        revocacao_esperada = 0.5
        precisao, revocacao = self.queryRunner.compute_precision_recall(
            3, lst_docs, relevant_docs)

        self.assertAlmostEqual(precisao, precisao_esperada,
                               places=2, msg=f"Valor incorreto de precisão")
        self.assertAlmostEqual(revocacao, revocacao_esperada,
                               places=2, msg=f"Valor incorreto de revocação")

    def check_terms_index(self, response: Mapping, expected_response: Mapping):
        # verifica se há algum termo faltando ou sobrando
        set_faltando = set(expected_response.keys())-set(response.keys())
        set_sobrando = set(response.keys()) - set(expected_response.keys())
        self.assertEqual(len(
            set_faltando), 0, f"Os seguintes termos deveriam ter sido retornados: {set_faltando}")
        self.assertEqual(len(
            set_sobrando), 0, f"Os seguintes termos não deveriam ter sido retornados: {set_sobrando}")

    def check_terms_occur(self, response: Mapping, expected_response: Mapping):
        # conversao para o codigo funcionar tanto quando os valores são uma Ocorrencia (no caso da query)
        # quanto for uma lista de correncia
        if len(response.values()) == 0:
            self.assertEqual(len(expected_response.values()), 0,
                             "O retorno da função deveria ser um dicionário vazio")
            return
        if type(list(response.values())[0]) != list:
            map_resp_list = {term: [occur] for term, occur in response.items()}
            map_expected_resp_list = {term: [occur]
                                      for term, occur in expected_response.items()}
        else:
            map_resp_list = response
            map_expected_resp_list = expected_response

        for term, lst_occur_resp in map_resp_list.items():
            for occur_resp in lst_occur_resp:
                bol_encontrou = False
                for expected_occur in map_expected_resp_list[term]:
                    if expected_occur.term_id == occur_resp.term_id and expected_occur.doc_id == occur_resp.doc_id:
                        self.assertEqual(expected_occur.term_freq, occur_resp.term_freq,
                                         f"A frequencia do termo '{term}' no documento {expected_occur.doc_id} deveria ser {expected_occur.term_freq}  e não {occur_resp.term_freq}")
                        bol_encontrou = True
                self.assertTrue(
                    bol_encontrou, msg=f"Não foi possível encontrar o termo '{term}' do documento {expected_occur.doc_id}")

    def test_get_query_term_occurence(self):
        arr_queries = ["crocodilo", "vocês",
                       "Vocês estejam", "vocês vocês crocodilo"]

        voces_id = self.index.get_term_id("vocês")
        estejam_id = self.index.get_term_id("estejam")

        arr_expected_response = [{},
                                 {
            "vocês": TermOccurrence(None, voces_id, 1),
        },
            {
            "vocês": TermOccurrence(None, voces_id, 1),
            "estejam": TermOccurrence(None, estejam_id, 1),

        },
            {
            "vocês": TermOccurrence(None, voces_id, 2)
        }

        ]

        for i, expected_response in enumerate(arr_expected_response):

            response = self.queryRunner.get_query_term_occurence(
                arr_queries[i])
            print(f"Consulta: {arr_queries[i]}")
            print(f"Resposta do método: {response}")
            self.check_terms_index(response, expected_response)
            print(f"Resposta esperada: {expected_response}")
            self.check_terms_occur(response, expected_response)
            print()

    def test_get_occurence_list_per_term(self):
        arr_terms = [["crocodilo"], ["vocês", "estejam", "crocodilo"]]

        voces_id = self.index.get_term_id("vocês")
        estejam_id = self.index.get_term_id("estejam")

        arr_expected_response = [{
            "crocodilo": []
        },
            {
            "vocês": [TermOccurrence(2, voces_id, 3), TermOccurrence(3, voces_id, 1)],
            "estejam":[TermOccurrence(3, estejam_id, 1)],
            "crocodilo":[]

        }
        ]
        for i, expected_response in enumerate(arr_expected_response):

            response = self.queryRunner.get_occurrence_list_per_term(
                arr_terms[i])
            print(f"Termos: {arr_terms[i]}")
            print(f"Resposta do método: {response}")
            self.check_terms_index(response, expected_response)
            self.check_terms_occur(response, expected_response)
            print("")

    def test_get_docs_term(self):
        arr_queries = ["crocodilo", "vocês",
                       "Vocês estejam", "vocês vocês crocodilo"]
        arr_expected_response = [[], [2, 3], [3, 2], [2, 3]]
        for i, query in enumerate(arr_queries):
            resposta, pesos = self.queryRunner.get_docs_term(query)
            print(f"Pesos dos documento da consulta '{query}': {pesos}")
            print()
            self.assertListEqual(
                resposta, arr_expected_response[i], f"A resposta a consulta '{query}' deveria ser {arr_expected_response[i]} e não {resposta}")

    def test_get_relevance_per_query(self):
        relevance_per_query = self.queryRunner.get_relevance_per_query()
        # print(relevance_per_query)

        expected_len = {
            "belo_horizonte": 27,
            "irlanda": 39,
            "sao_paulo": 605}
        for query in relevance_per_query.keys():
            resposta = len(relevance_per_query[query])
            self.assertEqual(
                resposta, expected_len[query],  f"A quantidade de documentos relevantes para '{query}' deveria ser {expected_len[query]} e não {resposta}")


if __name__ == "__main__":
    unittest.main()

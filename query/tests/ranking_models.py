from query.ranking_models import IndexPreComputedVals,VectorRankingModel,BooleanRankingModel,  OPERATOR
from index.structure import HashIndex,FileIndex,TermOccurrence
import unittest

class RankingModelTest(unittest.TestCase):
    def setUp(self):
        self.arr_indexes = [
                                {"a":[TermOccurrence(1,1,1),
                                    TermOccurrence(3,1,1)],
                                    "feia":[TermOccurrence(2,5,1),
                                            TermOccurrence(4,5,1)],
                                    "casa":[TermOccurrence(1,2,1),
                                            TermOccurrence(2,2,1),
                                            TermOccurrence(4,2,1),],
                                    "é":[TermOccurrence(1,3,1),
                                        TermOccurrence(2,3,1),
                                        TermOccurrence(3,3,1),
                                        TermOccurrence(4,3,1)],
                                    "verde":[TermOccurrence(1,4,2),
                                            TermOccurrence(3,4,1),
                                            TermOccurrence(4,4,1),],

                                     "velha":[TermOccurrence(3,6,1)]          
                                    }, 
                                {
                                    "new":[TermOccurrence(1, 1, 4),
                                        TermOccurrence(2, 1, 1),],
                                      "york":[TermOccurrence(1, 2, 1),TermOccurrence(2, 2, 1)],
                                     "post":[TermOccurrence(2, 3, 1)],
                                    "los":[TermOccurrence(3, 4, 1)],
                                    "angeles":[TermOccurrence(3, 5, 1)],
                                    "times":[TermOccurrence(1, 6, 1),TermOccurrence(3, 6, 1)],
                                }
                            ]

        self.arr_queries_per_idx = [[{"casa":TermOccurrence(None, 2, 1),
                                        "feia":TermOccurrence(None, 5, 1)},
                                      {"crocodilo":TermOccurrence(None, 2, 1)}
                                    ],
                                    [{"new":TermOccurrence(None, 1, 1),
                                    "times":TermOccurrence(None, 6, 1)}]
                                    ]
    def test_precomputed_vals(self):
        index = FileIndex()
        index.index("new",1,4)
        index.index("york",1,1)
        index.index("times",1,1)
        index.index("new",2,1)
        index.index("york",2,1)
        index.index("post",2,1)
        index.index("los",3,1)
        index.index("angeles",3,1)
        index.index("times",3,1)

        index.finish_indexing()

        precomp = IndexPreComputedVals(index)
        norma_esperada_per_doc = {1:1.94, 2:1.79, 3: 2.32}
        
        self.assertEqual(precomp.doc_count,3,"Numero de documentos inesperado")
        for doc_id,norma_esperada in norma_esperada_per_doc.items():
            self.assertAlmostEqual(norma_esperada, precomp.document_norm[str(doc_id)], places=2,msg=f"Norma inesperada do documento {doc_id}")
    def obtem_index_for_query(self,map_query,map_index):
        map_index_for_query = {}
        for term, list_ocur in map_index.items():
            if term in map_query.keys():
                map_index_for_query[term] = list_ocur
        return map_index_for_query

    def test_boolean_model(self):

        arr_set_esperado_and_per_query  = [[{2,4},set()],[{1}]]
        arr_set_esperado_or_per_query  = [[{1,2,4},set()],[{1,2,3}]]

        for idx, map_index in enumerate(self.arr_indexes):
            for query_position, map_query in enumerate(self.arr_queries_per_idx[idx]):
                model_and = BooleanRankingModel(OPERATOR.AND)
                map_index_for_query = self.obtem_index_for_query(map_query,map_index)
                lst_response,_ = model_and.get_ordered_docs(map_query, map_index_for_query)
                set_response =  set(lst_response)
                self.assertSetEqual(set_response, arr_set_esperado_and_per_query[idx][query_position],
                                    msg=f"Consulta com operador AND obteve um resultado inesperado ({set_response}) para o indice {idx} consulta {query_position}. Esperava-se: {arr_set_esperado_and_per_query[idx][query_position]} ")

                model_or = BooleanRankingModel(OPERATOR.OR)
                lst_response,_ = model_or.get_ordered_docs(map_query, map_index_for_query)
                set_response =  set(lst_response)
                self.assertSetEqual(set_response, arr_set_esperado_or_per_query[idx][query_position],
                                    msg=f"Consulta com operador OR obteve um resultado inesperado ({set_response}) para o indice {idx} consulta {query_position}. Esperava-se: {arr_set_esperado_or_per_query[idx][query_position]} ")
                


    def test_vector_model(self):
        index = FileIndex()
        precomp = IndexPreComputedVals(index)
        
        arr_lst_esperado_per_query  = [[[2,4,1],[]],[[1,2,3]]]
        peso_por_doc_esperado_per_query = [[{1:0.12,2:1.01,3:None,4:0.9},{}],
                                            [{1:0.709, 2: 0.19, 3:0.15}]]
        arr_norm_por_index = [{1:1.44,2:1.16,3:2.08,4:1.3},{1:1.93,2:1.78,3:2.31}]
        for idx, map_index in enumerate(self.arr_indexes):
            precomp.document_norm = arr_norm_por_index[idx]
            precomp.doc_count = len(arr_norm_por_index[idx].keys())
            for query_position, map_query in enumerate(self.arr_queries_per_idx[idx]):
                vector_model  = VectorRankingModel(precomp)
                map_index_for_query = self.obtem_index_for_query(map_query,map_index)
                lst_response, doc_weights = vector_model.get_ordered_docs(map_query, map_index_for_query)
                self.assertListEqual(lst_response, arr_lst_esperado_per_query[idx][query_position],
                                    msg=f"resposta não esperada para a consulta  {query_position} indice {idx}")
                
                for doc_id, peso in peso_por_doc_esperado_per_query[idx][query_position].items():
                    if doc_id not in doc_weights:
                        self.assertTrue(peso is None, f"O documento {doc_id} deveria ser recuperado da consulta {query_position} indice {idx}")
                    if peso is None:
                        self.assertTrue(doc_id not in doc_weights, f"O documento {doc_id} não deveria ser recuperado da consulta {query_position} indice {idx}")
                    else:
                        self.assertAlmostEqual(peso, doc_weights[doc_id], places=2,msg=f"Peso inesperado do documento {doc_id} consulta {query_position} índice {idx}. Peso calculado:{doc_weights[doc_id]} deveria ser: {peso}")
if __name__ == "__main__":
    unittest.main()
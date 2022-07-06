from index.structure import *

import unittest

class StructureTest(unittest.TestCase):
    def create_terms(self):
        #casa apareceu 10 vezes no doc. 1
        self.index.index("casa",1,10)
        #vermelho apareceu 3 vezes no doc. 1
        self.index.index("vermelho",1,3)
        #verde apareceu 1 vez no doc. 1
        self.index.index("verde",1,1)
        #vermelho apareceu 1 vez no doc. 2
        self.index.index("vermelho",2,1)
        #vermelho apareceu 1 vez no doc. 3
        self.index.index("vermelho",3,1)
        #casa apareceu 3 vezes no doc. 2
        self.index.index("casa",2,3)

        self.index.finish_indexing()

        print("======= Indice Gerado ======")
        print(self.index)

    def setUp(self):
        self.index = HashIndex()
        self.create_terms()

    def test_read_write(self):
        self.index.write("teste_idx.idx")
        
        idx_novo = Index.read("teste_idx.idx")

        #testa document count
        self.assertEqual(3,idx_novo.document_count)

        #testa a lista de ocorrencias
        self.occur_list_test(idx_novo)


    def test_document_count(self):
        self.assertEqual(3,self.index.document_count)

    def test_vocabulary(self):
        set_expected_vocab = {"casa","vermelho","verde"}
        set_vocab = self.index.vocabulary

        self.assertCountEqual(set_expected_vocab,set_vocab,f"Deveria haver a seguinte lista/conjunto: {set_expected_vocab} mas foi retornado: {set_vocab}")

    def test_document_count_with_term(self):
        self.assertEqual(2,self.index.document_count_with_term("casa"), f"Casa apareceu em dois documentos")
        self.assertEqual(3,self.index.document_count_with_term("vermelho"), f"Vemelho apareceu em dois documentos")
        self.assertEqual(1,self.index.document_count_with_term("verde"), f"Verde apareceu em dois documentos")
        self.assertEqual(0,self.index.document_count_with_term("cinza"), f"Cinza não está indexado, deveria retornar zero")

    def occur_list_test(self, index):
        dict_expected_index = {"casa":{1:10, 2:3},
                                "verde":{1:1},
                                "vermelho":{1:3,2:1,3:1}}

        for term, dic_doc_freq in dict_expected_index.items():
            list_occur = index.get_occurrence_list(term)
            for occur in list_occur:
                doc_id = occur.doc_id
                frequency = occur.term_freq
                self.assertTrue(doc_id in dict_expected_index[term], f"O termo {term} não ocorre no documento {doc_id}")
                self.assertEqual(dict_expected_index[term][doc_id],frequency, f"Erro o termo {term} deveria ocorrer {dict_expected_index[term][doc_id]}x no documento {doc_id} e não {frequency}")
            self.assertEqual( len(dic_doc_freq.keys()), len(list_occur), f"A lista de occorencia tem ocorrencias a mais (ou a menos) do termo {term}" )

        list_occur = index.get_occurrence_list('xuxu')
        self.assertListEqual(list_occur,[],"O termo xuxu não existe, deveria retornar lista vazia")

    def test_get_occurrence_list(self):
        self.occur_list_test(self.index)

class FileStructureTest(StructureTest):
    def setUp(self):
        self.index = FileIndex()
        self.create_terms()

if __name__ == "__main__":
    unittest.main()

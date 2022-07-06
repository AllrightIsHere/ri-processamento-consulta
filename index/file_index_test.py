from .structure import *
import unittest
from .index_structure_test import StructureTest
from .performance_test import PerformanceTest



class FileIndexTest(unittest.TestCase):

    def check_idx_file(self, obj_index, set_occurrences):
        #verifica a ordem das ocorrencias
        list_size = obj_index.idx_tmp_occur_last_element - obj_index.idx_tmp_occur_first_element + 1
        self.assertEqual(list_size,0,"A lista de ocorrencias deve ser zerada após chamar o método save_tmp_occurrences")
        last_occur = TermOccurrence(float('-inf'),float('-inf'),10)
        set_file_occurrences = set()
        with open(obj_index.str_idx_file_name,"rb") as idx_file:
            occur = obj_index.next_from_file(idx_file)
            while occur is not None:
                self.assertTrue(occur>last_occur, msg=f"A ocorrencia {last_occur} foi colocada de forma incorreta antes da ocorrencia {occur}")
                set_file_occurrences.add(occur)
                last_occur = occur
                occur = obj_index.next_from_file(idx_file)

        sobra_arquivo = set_file_occurrences-set_occurrences
        sobra_lista = set_occurrences-set_file_occurrences
        self.assertEqual(len(sobra_arquivo),0, f"Existem ocorrências no arquivo que não estavam na 'lst_occurrences_tmp': {sobra_arquivo} ")
        self.assertEqual(len(sobra_lista),0, f"As seguintes ocorrências não foram inseridas no arquivo de indice: {sobra_lista} ")

    def test_next_from_file(self):
        self.index = FileIndex()
        occur1 = TermOccurrence(2,1,5)
        occur2 = TermOccurrence(10,2,1)
        with open("term_test","wb") as idx_new_file:
            occur1.write(idx_new_file)
            occur2.write(idx_new_file)

        with open("term_test","rb") as file:
            occur1_read = self.index.next_from_file(file)
            occur2_read = self.index.next_from_file(file)
            occur3_read = self.index.next_from_file(file)
            self.assertEqual(occur1,occur1_read, f"Primeiro elemento deveria ser {occur1} porém foi obtido {occur1_read}")
            self.assertEqual(occur2,occur2_read, f"Segundo elemento deveria ser {occur2} porém foi obtido {occur2_read}")
            self.assertIsNone(occur3_read,"Não há 3o elemento, assim, deveria retornar None na terceira leitura")
    
    def test_next_from_list(self):
        self.index = FileIndex()
        #testa o size
        self.index.idx_tmp_occur_last_element  = -1
        self.index.idx_tmp_occur_first_element = 0
        self.assertEqual(self.index.get_tmp_occur_size(),0, f"Tamanho incorreto da lista")


        self.index.add_index_occur(None, 2, 1, 5)
        self.index.add_index_occur(None, 3, 1, 1)
        self.index.add_index_occur(None, 1, 2, 1)
        self.index.add_index_occur(None, 3, 2, 4)
        terms_to_add = [TermOccurrence(2,1,5),
                                        TermOccurrence(3,1,1),
                                        TermOccurrence(1,2,1),
                                        TermOccurrence(3,2,3)]
        
        self.assertEqual(self.index.get_tmp_occur_size(),4, f"Tamanho incorreto da lista")
        
        for i in range(4):
            next_term = self.index.next_from_list()
            self.assertEqual(next_term, terms_to_add[i], f"next_from_list deveria remover o termo {terms_to_add[i]} e foi o {next_term} ")
            self.assertEqual(self.index.get_tmp_occur_size(), 3-i, f"Tamanho incorreto da lista ao remover o elemento #{i}")
        next_term = self.index.next_from_list()
        self.assertEqual(self.index.get_tmp_occur_size(), 3-i, f"Após a remoção de todos os elementos da lista, o tamanho deveria ser vazio")
        self.assertIsNone(next_term,"Após a remoção de todos os elementos da lista, o metodo next_from_list deveria retornar none")

    def test_save_tmp_occurrences(self):

        #testa a primeira vez (adicionando tudo na primeira vez)
        self.index = FileIndex()
        set_occurrences = []
        self.index.lst_occurrences_tmp = [TermOccurrence(2,4,5),
                                        TermOccurrence(2,2,1),
                                        TermOccurrence(1,2,1),
                                        TermOccurrence(1,1,3),
                                        None,
                                        None,
                                        None,
                                        None]
        set_occurrences = set(self.index.lst_occurrences_tmp) - {None}
        self.index.idx_tmp_occur_last_element  = 3
        self.index.save_tmp_occurrences()
        self.check_idx_file(self.index, set_occurrences)
        print("Primeira execução (criação inicial do indice) [ok]")

        #adicina alguns
        self.index.lst_occurrences_tmp = [TermOccurrence(1,3,3),
                                        TermOccurrence(2,3,4)]
        self.index.idx_tmp_occur_last_element  = 1
        set_occurrences = set_occurrences | set(self.index.lst_occurrences_tmp)
        self.index.save_tmp_occurrences()
        self.check_idx_file(self.index, set_occurrences)
        print("Inserção de alguns itens - teste 1/2 [ok]")




        #adiciona mais alguns
        self.index.lst_occurrences_tmp = [TermOccurrence(2,1,2),
                                        TermOccurrence(3,2,2),
                                        TermOccurrence(3,1,1)]
        self.index.idx_tmp_occur_last_element  = 2
        #checa ordenação do arquivo e verifica todas as ocorrencias existem
        set_occurrences = set_occurrences|set(self.index.lst_occurrences_tmp)
        self.index.save_tmp_occurrences()
        self.check_idx_file(self.index, set_occurrences)
        print("Inserção de alguns itens - teste 2/2 [ok]")

    def test_finish_indexing(self):
        self.index = FileIndex()
        self.index.idx_tmp_occur_last_element  = 8
        self.index.idx_tmp_occur_first_element = 0
        self.index.lst_occurrences_tmp = [
                                        TermOccurrence(1,1,3),
                                        TermOccurrence(1,2,1),
                                        TermOccurrence(1,3,3),
                                        TermOccurrence(1,4,5),
                                        TermOccurrence(2,1,2),
                                        TermOccurrence(2,2,1),
                                        TermOccurrence(2,4,5),
                                        TermOccurrence(3,1,1),
                                        TermOccurrence(3,2,2),
                                        None,
                                        None,
                                        None,
                                        None
                                        ]


        print("Lista de ocorrências a serem testadas:")
        for i,occ in enumerate(self.index.lst_occurrences_tmp):
            print(f"{occ}")
        x = 100
        int_size_of_occur = None
        with open("teste_file.idx","wb") as file:
            self.index.lst_occurrences_tmp[0].write(file)
            int_size_of_occur = file.tell()

        print(f"Tamanho de cada ocorrência: {int_size_of_occur} bytes")
        self.index.save_tmp_occurrences()
        #verifica, para cada posição
        self.index.dic_index = {"casa":TermFilePosition(1),
                                "verde":TermFilePosition(2),
                                "prédio":TermFilePosition(3),
                                "amarelo":TermFilePosition(4)}
        self.index.finish_indexing()

        arr_termos = ["casa","verde","prédio","amarelo"]
        #testa se o id manteve-se o mesmo
        [self.assertEqual(self.index.dic_index[arr_termos[i]].term_id,i+1,f"O id do termo {i+1} mudou para {self.index.dic_index[arr_termos[i]].term_id}") for i in range(4)]



        #testa se a quantidade de documentos que possuem um determinado termo está correto
        arr_pos_por_termo = [0,int_size_of_occur*3,int_size_of_occur*6,int_size_of_occur*7]
        arr_pos = [1,4,7,8]
        [self.assertEqual(self.index.dic_index[arr_termos[i]].term_file_start_pos,arr_pos_por_termo[i],f"A posição inicial do termo de id {i+1} no arquivo seria {arr_pos_por_termo[i]} (ou seja, antes da {arr_pos[i]}ª ocorrencia) e não {self.index.dic_index[arr_termos[i]].term_file_start_pos}") for i in range(4)]

        #testa se a quantidade de documentos que possuem um determinado termo está correto
        arr_doc_por_termo = [3,3,1,2]
        [self.assertEqual(self.index.dic_index[arr_termos[i]].doc_count_with_term,arr_doc_por_termo[i],f"A quantidade de documentos que possuem o termo de id {self.index.dic_index[arr_termos[i]].term_id} seria {arr_doc_por_termo[i]} e não {self.index.dic_index[arr_termos[i]].doc_count_with_term}") for i in range(4)]





if __name__ == "__main__":
    unittest.main()

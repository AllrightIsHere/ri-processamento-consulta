from copyreg import pickle
from typing import List, Set, Mapping
from nltk.tokenize import word_tokenize
from numpy import append
from sqlalchemy import true
from util.time import CheckTime
from query.ranking_models import RankingModel, BooleanRankingModel, VectorRankingModel, IndexPreComputedVals
from index.structure import Index, TermOccurrence
from index.indexer import Cleaner, HTMLIndexer
import datetime
import os
import pickle


class QueryRunner:
    def __init__(self, ranking_model: RankingModel, index: Index, cleaner: Cleaner):
        self.ranking_model = ranking_model
        self.index = index
        self.cleaner = cleaner

    @staticmethod
    def get_relevance_per_query() -> Mapping[str, Set[int]]:
        """
        Adiciona a lista de documentos relevantes para um determinada query (os documentos relevantes foram
        fornecidos no ".dat" correspondente. Por ex, belo_horizonte.dat possui os documentos relevantes da consulta "Belo Horizonte"

        """
        dic_relevance_docs = {}
        for arquiv in ["belo_horizonte", "irlanda", "sao_paulo"]:
            with open(f"relevant_docs/{arquiv}.dat") as arq:
                arr = [int(s) for s in arq.readline().split(",")]
                dic_relevance_docs[arquiv] = set(arr)
                # print(dic_relevance_docs)
        return dic_relevance_docs

    def count_topn_relevant(self, n: int, respostas: List[int], doc_relevantes: Set[int]) -> int:
        """
        Calcula a quantidade de documentos relevantes na top n posições da lista lstResposta que é a resposta a uma consulta
        Considere que respostas já é a lista de respostas ordenadas por um método de processamento de consulta (BM25, Modelo vetorial).
        Os documentos relevantes estão no parametro docRelevantes
        """
        # print(f"Respostas: {respostas} doc_relevantes: {doc_relevantes}")
        relevance_count = 0

        for i in range(n):
            if i >= len(respostas):
                break

            elif respostas[i] in doc_relevantes:
                relevance_count += 1

        return relevance_count

    def compute_precision_recall(self, n: int, lst_docs: List[int], relevant_docs: Set[int]) -> (float, float):
        count_relevant = self.count_topn_relevant(n, lst_docs, relevant_docs)
        precision = float(count_relevant / min(len(lst_docs), n))
        recall = float(count_relevant / len(relevant_docs))
        return precision, recall

    def get_query_term_occurence(self, query: str) -> Mapping[str, TermOccurrence]:
        """
                Preprocesse a consulta da mesma forma que foi preprocessado o texto do documento (use a classe Cleaner para isso).
                E transforme a consulta em um dicionario em que a chave é o termo que ocorreu
                e o valor é uma instancia da classe TermOccurrence (feita no trabalho prático passado).
                Coloque o docId como None.
                Caso o termo nao exista no indic, ele será desconsiderado.
        """
        # print(self.index)
        map_term_occur = {}
        dic_word_count = {}

        # print(query)

        plain_text = self.cleaner.preprocess_text(query)

        tokens = word_tokenize(plain_text, language="portuguese")

        for token in tokens:
            term = self.cleaner.preprocess_word(token)

            if term is not None:
                if term not in dic_word_count:
                    dic_word_count[term] = 0
                dic_word_count[term] += 1

        # print("1", dic_word_count)

        for term, count in dic_word_count.items():
            if self.index.get_occurrence_list(term):
                map_term_occur[term] = TermOccurrence(
                    doc_id=None, term_freq=count, term_id=self.index.get_term_id(term))

        return map_term_occur

    def get_occurrence_list_per_term(self, terms: List) -> Mapping[str, List[TermOccurrence]]:
        """
                Retorna dicionario a lista de ocorrencia no indice de cada termo passado como parametro.
                Caso o termo nao exista, este termo possuirá uma lista vazia
        """

        dic_terms = {}

        plain_text = self.cleaner.preprocess_text(" ".join(terms))

        tokens = word_tokenize(plain_text, language="portuguese")

        for term in tokens:
            term = self.cleaner.preprocess_word(term)
            # print("4", term)
            dic_terms[term] = self.index.get_occurrence_list(term)

        return dic_terms

    def get_docs_term(self, query: str) -> List[int]:
        """
                A partir do indice, retorna a lista de ids de documentos desta consulta
                usando o modelo especificado pelo atributo ranking_model
        """

        query = query.replace("_", " ")

        # Obtenha, para cada termo da consulta, sua ocorrencia por meio do método get_query_term_occurence
        dic_query_occur = self.get_query_term_occurence(query)

        # obtenha a lista de ocorrencia dos termos da consulta
        dic_occur_per_term_query = self.get_occurrence_list_per_term(
            query.split(" "))

        # print("2", dic_query_occur)
        # print("3", dic_occur_per_term_query)
        # utilize o ranking_model para retornar o documentos ordenados considrando dic_query_occur e dic_occur_per_term_query
        return self.ranking_model.get_ordered_docs(query=dic_query_occur, docs_occur_per_term=dic_occur_per_term_query)

    @staticmethod
    def runQuery(query: str, indice: Index, indice_pre_computado: IndexPreComputedVals, map_relevantes: Mapping[str, Set[int]]):
        """
                Para um daterminada consulta `query` é extraído do indice `index` os documentos mais relevantes, considerando 
                um modelo informado pelo usuário. O `indice_pre_computado` possui valores précalculados que auxiliarão na tarefa. 
                Além disso, para algumas consultas, é impresso a precisão e revocação nos top 5, 10, 20 e 50. Essas consultas estão
                Especificadas em `map_relevantes` em que a chave é a consulta e o valor é o conjunto de ids de documentos relevantes
                para esta consulta.
        """
        time_checker = CheckTime()

        # PEça para usuario selecionar entre Booleano ou modelo vetorial para intanciar o QueryRunner
        # apropriadamente. NO caso do booleano, vc deve pedir ao usuario se será um "and" ou "or" entre os termos.
        # abaixo, existem exemplos fixos.
        cleaner = Cleaner(stop_words_file="stopwords.txt",
                          language="portuguese",
                          perform_stop_words_removal=True,
                          perform_accents_removal=True,
                          perform_stemming=False)

        query = cleaner.preprocess_text(query)

        query = "_".join(query.split(" "))

        model_select = int(
            input('Qual modelo vc deseja?\n (1) Booleano\n (2) Vetorial\n  Sua resposta: '))
        bool_selected = 1
        if model_select == 1:
            bool_selected = int(
                input('Qual operação booleana?\n (1) AND\n (2) OR\n  Sua resposta: '))

        ranking_model = VectorRankingModel(
            indice_pre_computado) if model_select == 2 else BooleanRankingModel(bool_selected)


        qr = QueryRunner(
            index=indice, ranking_model=ranking_model, cleaner=cleaner)
        time_checker.print_delta("Query Creation")

        # Utilize o método get_docs_term para obter a lista de documentos que responde esta consulta

        resposta = qr.get_docs_term(query)
        # print(resposta)
        resposta = resposta[0]
        if model_select == 1:
            resposta = list(resposta)
        time_checker.print_delta(f"anwered with {len(resposta)} docs")

        # nesse if, vc irá verificar se o termo possui documentos relevantes associados a ele
        # se possuir, vc deverá calcular a Precisao e revocação nos top 5, 10, 20, 50.
        # O for que fiz abaixo é só uma sugestao e o metododo countTopNRelevants podera auxiliar no calculo da revocacao e precisao
        # print(map_relevantes)
        # map_relevantes['belo horizonte'] = map_relevantes.pop('belo_horizonte')
        # map_relevantes['sao paulo'] = map_relevantes.pop('sao_paulo')
        if query in map_relevantes:
            relevants_list = list(map_relevantes[query])
            if any(relevant_doc in resposta for relevant_doc in relevants_list):
                doc_relevantes = map_relevantes[query]
                arr_top = [5, 10, 20, 50]
                revocacao = 0
                precisao = 0
                for n in arr_top:
                    precisao, revocacao = qr.compute_precision_recall(
                        n, resposta, doc_relevantes)
                    print(f"Precisao @{n}: {precisao}")
                    print(f"Recall @{n}: {revocacao}")

        # imprima aas top 10 respostas
        top_respostas = resposta[:10]
        for i in range(len(top_respostas)):
            print(f"{i+1}: {top_respostas[i]}")

    @staticmethod
    def main():
        # leia o indice (base da dados fornecida)
        index: Index = Index.read("wiki.idx")

        # Checagem se existe um documento (apenas para teste, deveria existir)
        print(f"Existe o doc? index.hasDocId(105047)")

        # Instancie o IndicePreCompModelo para pr ecomputar os valores necessarios para a query
        print("Precomputando valores atraves do indice...")
        check_time = CheckTime()
        if os.path.exists('pre_compute.idx'):
            with open('pre_compute.idx', 'rb') as f:
                index_pre_compute = pickle.load(f)
        else:
            index_pre_compute = IndexPreComputedVals(index)
            with open('pre_compute.idx', 'wb') as f:
                pickle.dump(index_pre_compute, f)
        check_time.print_delta("Precomputou valores")

        # encontra os docs relevantes
        map_relevance = QueryRunner.get_relevance_per_query()

        # aquui, peça para o usuário uma query (voce pode deixar isso num while ou fazer uma interface grafica se estiver bastante animado ;)
        query = "São Paulo"
        while true:
            print("===========================================")
            opcao = int(
                input('Voce deseja fazer uma consulta?\n (1) SIM\n (2) NAO\n  Sua resposta: '))
            if(opcao == 1):
                query = input('  Digite a query: ')
                print(f"Fazendo query de '{query}'...")
                QueryRunner.runQuery(
                    query, index, index_pre_compute, map_relevance)
                print("===========================================")
            else:
                print("Finalizando consultas. :)")
                print("===========================================")
                break

from index.indexer import *
from index.structure import *
import unittest

class IndexerTest(unittest.TestCase):
    def test_indexer(self):
        obj_index = FileIndex()
        html_indexer = HTMLIndexer(obj_index)
        html_indexer.index_text_dir("index/docs_test")
        set_vocab = set(obj_index.vocabulary)
        set_expected_vocab = set([ 'cas', 'ser', 'verd', 'ou', 'nao', 'eis', 'questa'])

        sobra_expected = set_expected_vocab-set_vocab
        sobra_vocab = set_vocab-set_expected_vocab

        self.assertTrue(len(sobra_expected) == 0 and len(sobra_vocab)==0, f"O Vocabulário indexado não é o esperado!\nVocabulario indexado: {set_vocab}\nVocabulário esperado: {set_expected_vocab}")
        lst_occur = obj_index.get_occurrence_list("cas")
        dic_expected = {111:TermOccurrence(111,2,1),
                        100102:TermOccurrence(100102,2,2)}
        for occur in lst_occur:
                self.assertTrue(type(occur.doc_id) == int,f"O tipo do documento deveria ser inteiro")
                self.assertTrue(occur.doc_id in dic_expected,f"O docid número {occur.doc_id} não deveria existir ou não deveria indexar o termo 'cas'")
                self.assertEqual(dic_expected[occur.doc_id].term_freq,occur.term_freq, f"A frequencia do termo 'cas' no documento {occur.doc_id} deveria ser {occur.term_freq}")
    
    def test_wiki_idx(self):
        wiki_idx = Index.read("wiki.idx")

        self.assertTrue(wiki_idx.document_count>60000)
        self.assertEqual(len(wiki_idx.get_occurrence_list("casa")), 4632)
        self.assertEqual(len(wiki_idx.get_occurrence_list("belo")), 474)
        self.assertEqual(len(wiki_idx.get_occurrence_list("horizonte")), 234)
if __name__ == "__main__":
    unittest.main()

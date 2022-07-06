from IPython.display import clear_output
from index.structure import *

from datetime import datetime
import math
import tracemalloc
import unittest
from random import randrange,seed

from util.performance import CheckPerformance




class PerformanceTest(unittest.TestCase):
    NUM_DOCS = 2000
    NUM_TERM_PER_DOC = 500

    def setUp(self):
        self.index = HashIndex()
        self.perfomance = CheckPerformance()

    def create_vocabulary(self):
        vocabulary = []
        for i in range(65,91):
            for j in range(65,91):
                for k in range(65,91):
                    vocabulary.append(f"{chr(i)}{chr(j)}{chr(k)}")
        return vocabulary


    def index_words(self):
        count = 0
        total = PerformanceTest.NUM_DOCS*PerformanceTest.NUM_TERM_PER_DOC
        self.perfomance = CheckPerformance(count_total=total, clear_output=True)
        seed(10)
        words = []
        for doc_i in range(PerformanceTest.NUM_DOCS):
            for term_j in range(PerformanceTest.NUM_TERM_PER_DOC):
                idx_term = randrange(0,len(self.vocabulary))
                str_term = self.vocabulary[idx_term]
                self.index.index(str_term, doc_i, (count%10)+1)
                #indiceTeste.index(vocabulario[(count+1)%15625], d, (count%10)+1);
                if count%10000==0:
                    self.perfomance.print_step("Indexação de termo", count)

                count+=1
        return count

    def test_performance(self):


        self.vocabulary = self.create_vocabulary()

        total = self.index_words()


import time
class FilePerformanceTest(PerformanceTest):
    def setUp(self):
        self.index = FileIndex()

def test():
    for i in range(10):
        clear_output(wait=True)
        print(f"oi {i}")
        time.sleep(1)

if __name__ == "__main__":
    unittest.main()

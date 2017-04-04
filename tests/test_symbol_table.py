import unittest
import sys
from scipy import random, linalg
import numpy as np
from sydar.symbol_table import *
from sydar.parser import *

class ParserFuncTest(unittest.TestCase):
    
    def setUp(self):
        self.symb = parse_miu('examples/input.miu')

    def tearDown(self):
        pass

    def test_get_tagged_nodes(self):
        nodes = self.symb.get_tagged_nodes()
        self.assertIsInstance(nodes, dict)

    def test_get_tagged_edges(self):
        edges = self.symb.get_tagged_edges()
        self.assertIsInstance(edges, dict)

class ParserClassTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
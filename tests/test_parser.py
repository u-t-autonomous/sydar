import unittest
import sys
# sys.path.append("..")
# from parser import *
from scipy import random, linalg
import numpy as np
# from symbol_table import SymbolTable
from sydar import parser, symbol_table

class ParserFuncTest(unittest.TestCase):
	
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parse_imu(self):
        self.assertIsInstance(parser.parse_miu('examples/input.miu'), symbol_table.SymbolTable)

    def test_check_letter(self):
        failed = False
        try:
            parser.parse_miu('examples/bad_input.miu')
        except SystemExit as e:
            if e.code == 'check_letter':
                failed = True
        self.assertTrue(failed)

class ParserClassTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
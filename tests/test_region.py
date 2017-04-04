import unittest
import sys
# from region import *
from scipy import random, linalg
import numpy as np
from random import shuffle
# from utils import *
# from helpers import *
import sydar
from sydar.helpers import *

class RegionFuncTest(unittest.TestCase):

    def setUp(self):
        pd1 = generate_pd(2)
        point1 = generate_point(2)
        self.ellipsoid1 = Ellipsoid(pd1,point1)
        pd2 = generate_pd(2)
        point2 = generate_point(2)
        self.ellipsoid2 = Ellipsoid(pd2,point2)
        self.tree = Tree('||',[self.ellipsoid1,self.ellipsoid2])

    def tearDown(self):
        pass

    def test_over_approx(self):
        ellip = self.tree.over_approx()
        #self.assertIsInstance(ellip,Ellipsoid)

    def test_under_approx(self):
        self.assertTrue(True)

    def test_tag_nodes(self):
        self.assertTrue(True)

    def test_plot_tree(self):
        self.ellipsoid1.show()

class RegionClassTest(unittest.TestCase):

    def setUp(self):
        self.pd = generate_pd(5)
        self.point = generate_point(5)
        self.vector = generate_point(5)
        self.ellipsoid = Ellipsoid(self.pd,self.point)
        self.hs = HalfSpace(self.point,self.vector)
        self.empty = Empty()
        self.tree = Tree('||',[self.hs,self.empty,self.ellipsoid])

    def tearDown(self):
        pass

    def test_Terminal(self):
        self.assertTrue(True)

    def test_Tree(self):
        self.assertIsInstance(self.tree, Tree)

    def test_HalfSpace(self):
        self.assertIsInstance(self.hs, HalfSpace)

    def test_Empty(self):
        self.assertIsInstance(self.empty, Empty)

    def test_Ellipsoid(self):
        self.assertIsInstance(self.ellipsoid, Ellipsoid)

if __name__ == '__main__':
    unittest.main()
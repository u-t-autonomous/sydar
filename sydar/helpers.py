"""
.. module:: helpers
   :platform: Unix
   :synopsis: 

.. moduleauthor:: Mohammed Alshiekh


"""

from region import *
from scipy import random

def generate_psd(size):
    matrixSize = size
    A = random.rand(matrixSize,matrixSize)
    psd = np.dot(A,A.transpose())
    return psd.tolist()

def generate_pd(size):
    psd = generate_psd(size)
    alpha = 0.1
    I = np.identity(size)
    pd = psd + alpha*I
    return pd.tolist()

def generate_point(size,shift=0):
    matrixSize = size
    point = random.rand(matrixSize,1)+shift
    return point[:,0].tolist()

def generate_ellipsoid(size,shift=0):
    pd = generate_pd(2)
    point = generate_point(2,shift)
    return Ellipsoid(pd,point)    

def generate_halfspace(size, shift=0):
    point = generate_point(2,shift)
    vector = generate_point(2,shift)
    return HalfSpace(point,vector)

def generate_canon_tree(n_regions=5, n_operations=2,regions=None):
    sub_tree1 = Tree('&',[x for x in regions[:2]])
    sub_tree2 = Tree('&',[x for x in regions[2:4]])
    sub_tree3 = Tree('&',[x for x in regions[4:6]])
    sub_trees = [sub_tree1, sub_tree2, sub_tree3]
    return Tree('||',sub_trees)

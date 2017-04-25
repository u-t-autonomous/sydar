import numpy as np
import sys

def is_symmetric(matrix):
    return (matrix.transpose() == matrix).all()

def is_pos_def(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)

def is_pos_semidef(matrix):
    return np.all(np.linalg.eigvals(matrix) >= 0)

def is_point(coord):
    return (len(np.array(coord).shape) == 1) or np.array(coord).shape[0]==1 or np.array(coord).shape[0]==2

def is_PD(P):
    return isinstance(P, PDMatrix)

def is_center_point(c):
    return isinstance(c, CenterPoint)

def dim_check(P,c):
    return P.shape[0] == c.shape[0]

class Objective(object):
    def __init__(self, p, q, r):
        self.p = p
        self.q = q
        self.r = r
        
    def __str__(self):
        return 'Objective({}, {}, {})'.format(self.p, self.q, self.r)  

    def __repr__(self):
        return 'Objective({}, {}, {})'.format(self.p, self.q, self.r)  

class Matrix(object):
    def __init__(self, A):
        self.matrix = np.array(A)
        self.shape = self.matrix.shape
        
    @property
    def matrix(self):
        """I'm the 'matrix' property."""
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value
    
    def __str__(self):
        return '{}'.format(self.matrix)       

    @matrix.deleter
    def matrix(self):
        del self._matrix       
 
    def mmat(self, format='%.4e'):
        """Display the ndarray 'x' in a format suitable for pasting to MATLAB"""
        string = ''

        def print_row(row, format):
            sub_string = ''
            for i in row:
                sub_string += format % i + ' '
            return sub_string

        if self._matrix.ndim == 1:
            # 1d input
            string += "["
            string += print_row(self._matrix, format)
            string += "];\n"
            string += "\n"
            string = string[:-3]

        if self._matrix.ndim == 2:
            string += "["
            string += print_row(self._matrix[0], format)
            if self._matrix.shape[0] > 1:
                string += ';'
            for row in self._matrix[1:-1]:
                string += " "
                string += print_row(row, format)
                string += ";"
            if self._matrix.shape[0] > 1:
                string += " "
                string += print_row(self._matrix[-1], format)
            string += "];"
            string = string[:-1]

        if self._matrix.ndim > 2:
            d_to_loop = self._matrix.shape[2:]
            sls = [slice(None,None)]*2
            string += "reshape([ "
            # loop over flat index
            for i in range(prod(d_to_loop)):
                # reverse order for matlab
                # tricky double reversal to get first index to vary fastest
                ind_tuple = unravel_index(i,d_to_loop[::-1])[::-1]
                ind = sls + list(ind_tuple)
                string += mmat(self._matrix[ind],format)+"\n"         

            string += '],['
            for i in self._matrix.shape:
                string += '%d' % i
            string += '])'+"\n" 
        return string

class SymMatrix(Matrix):
    @Matrix.matrix.setter
    def matrix(self, value):
        if is_symmetric(value):
            self._matrix = value
        else:
            print "The matrix is not a symmetric matrix!"
            sys.exit(1)  

class PDMatrix(Matrix):
    @Matrix.matrix.setter
    def matrix(self, value):
        if is_pos_def(value):
            self._matrix = value
        else:
            print "The matrix is not a PD matrix!"
            sys.exit(1)          

class PSDMatrix(Matrix):
    @Matrix.matrix.setter
    def matrix(self, value):
        if is_pos_semidef(value):
            self._matrix = value
        else:
            print "The matrix is not a PSD matrix!"
            sys.exit(1)

class CenterPoint(object):
    def __init__(self, point):
        self.coord = np.array(point)
        self.shape = self.coord.shape
    @property
    def coord(self):
        """I'm the 'coord' property."""
        return self._coord

    @coord.setter
    def coord(self, value):
        if is_point(np.transpose(value)):
            self._coord = value
        else:
            print "The input must be a point, the given is {}".format(value)
            sys.exit(1)

    @coord.deleter
    def coord(self):
        del self._coord   

    def __str__(self):
        return '{}'.format(self.coord)                    
         
        
# def main():
#     c = CenterPoint([1,2,5])
#     h = HalfSpace([1,2,3],[4,3,8], 'h')
#     a = PDMatrix([[1,0,0],[0,1,0],[0,0,1]])
#     k = Matrix([[1,0,0],[0,1,0],[0,0,1]])
#     ellipsoid = Ellipsoid(a,c,'A')
#     print ellipsoid
#     print h
#     print a.mmat()

# if __name__ == '__main__':
#     main()
"""
.. module:: region
   :platform: Unix
   :synopsis: 

.. moduleauthor:: Mohammed Alshiekh

This module contains many functions that are developed in the ast 
module for TuLiP and can be found here:
https://github.com/tulip-control/tulip-control/blob/master/tulip/spec/ast.py


"""

from abc import ABCMeta, abstractmethod
from utils import *
from copy import deepcopy
from random import random

OPMAP = {'||': '||', '&': '&','!': '!'}

def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5 ):
    '''If there is a cycle that is reachable from root, then result 
       will not be a hierarchy.

       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - 
       avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
    '''
    neighbors = G.neighbors(root)
    def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,
     poss = None, parent = None, parsed = [] ):
        if(root not in parsed):
            parsed.append(root)
            if poss == None:
                poss = {root:(xcenter,vert_loc)}
            else:
                poss[root] = (xcenter, vert_loc)
            neighbors = G.neighbors(root)
            if len(neighbors)!=0:
                dx = width/len(neighbors) 
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    poss = h_recur(G,neighbor, width = dx, vert_gap = vert_gap,
                            vert_loc = vert_loc-vert_gap, xcenter=nextx, 
                            poss=poss, parent = root, parsed = parsed)
        return poss

    return h_recur(G, root, width=1., vert_gap = 0.2, 
        vert_loc = 0, xcenter = 0.5)

class RegionNode(object):
    """Base class for all nodes."""

    __metaclass__ = ABCMeta
    opmap = OPMAP

    @abstractmethod
    def __init__(self):
        pass

    #@abstractmethod
    #def __repr__(self):
    #    pass

    @abstractmethod
    def flatten(self):
        pass

    def __and__(self, other):
        return Tree('&', [self, other])

    def __or__(self, other):
        return Tree('||', [self, other])

    def complement(self):
        return Tree('!', self)

    def over_approx(self):
        return RegionNode()

    def over_approx(self):
        return RegionNode()  
    
    def __eq__(self, other):
        return self.flatten() == other.flatten()
    
    def empty_clear_complete(self):
        return True      

    def clear_single_complete(self):
        """
        Checks if a tree has any single leafed nodes. 
        If yes, it returns False, else True.
        """
        return True

class Terminal(RegionNode):
    """Base class for all terminal nodes."""
    def __init__(self, approx=False):
        self.approx = approx

    def insert_intersection(self):
        return Tree('&', [self])        

    def redund_clear_complete(self):
        """
        Checks if a tree has any redundant nodes. 
        If yes, it returns False, else True.
        """
        return True

    def to_canon_tree(self):
        node = deepcopy(self)
        tree = Tree('||',[Tree('&',[node], level = 2)], level = 1)
        return tree
    
class Tree(RegionNode):
    def __init__(self, operator, operands, level = None):
        try:
            operator + 'a'
        except TypeError:
            raise TypeError(
                'operator must be string, got: {op}'.format(
                    op=operator))
        self.type = 'operator'
        self.operator = operator
        self.operands = list(operands)
        self.level = level

    @property
    def operator(self):
        """I'm the '_operator' property."""
        return self._operator

    @operator.setter
    def operator(self, value):
        if value in self.opmap:
            self._operator = value
        else:
            print "The operator is not part of the OPMAP: {}".format(
                " ".join(self.opmap))
            sys.exit(1)

    @operator.deleter
    def operator(self):
        del self._operator          

    def __repr__(self):
        return self.flatten()
        return '{t}({op}, {xyz})'.format(
            t=type(self).__name__,
            op=repr(self.operator),
            xyz=', '.join(repr(x) for x in self.operands))

    def __str__(self, depth=None):
        return self.flatten_infix()
        if depth is not None:
            depth = depth - 1
        if depth == 0:
            return '...'
        return '({op},{xyz})'.format(
            op=self.operator,
            xyz=','.join(x.__str__(depth=depth)
                         for x in self.operands))

    def __len__(self):
        return 1 + sum(len(x) for x in self.operands)

    def flatten(self):
        return ' '.join([
            '(',
            self.opmap[self.operator],
            ', '.join(x.flatten() for x in self.operands),
            ')'])  

    def flatten_infix(self):
        return ' '.join([
            '(',
            '{} '.join(x.flatten() for x in self.operands).format(self.opmap[self.operator]),
            ')'])  
    
    def insert_intersection(self):
        if self.operator == '&':
            self.level = 2
            return self
        else:
            return Tree('&', [self], level = 2)
    
    def show(self):
        import networkx as nx
        import matplotlib.pyplot as plt
        self.colors = []
        self.node_counter = 2
        self.G = nx.Graph()
        self.labels = {}
        self.G.add_node(1)
        self.colors.append('g')
        self.labels[1] = self.operator
        self.parse(1)
        poss = hierarchy_pos(self.G, root=1)  
        nx.draw(self.G, pos=poss, with_labels=False)
        nx.draw_networkx_nodes(self.G, pos=poss, node_color=self.colors)
        self.G = nx.relabel_nodes(self.G, self.labels)
        nx.draw_networkx_labels(self.G, pos=poss, labels=self.labels) 
        plt.show()
        
    def parse(self, parent, region=None):
        if region is None:
            region = self
        for operand in region.operands:
            if isinstance(operand, Terminal):
                # Add node
                self.G.add_node(self.node_counter)
                self.colors.append('r' if operand.approx else 'g')
                self.G.add_edge(parent, self.node_counter)
                #print("Added edge from", parent, "to", self.node_counter)
                self.labels[self.node_counter] = type(operand).__name__
                self.node_counter += 1
            elif isinstance(operand, RegionNode):
                # Add sub-tree
                self.G.add_node(self.node_counter)
                self.colors.append('g')
                self.G.add_edge(parent, self.node_counter)
                self.labels[self.node_counter] = operand.operator
                self.node_counter += 1
                self.parse(self.node_counter-1, operand)
            else:
                raise ValueError('Incorrect operand found by parse()')

    def to_canon_tree(self):
        """
        This routine transforms any tree to a canonical tree.
        """
        tree = deepcopy(self)
        done = False
        while not done:
            tree = tree.extend_tree()
            tree = tree.lossless_transform()
            done = True
        #tree = tree.lossy_transform()
        return tree

    def extend_tree(self):
        tree = deepcopy(self)
        if tree.operator != '||':
            tree = Tree('||',[tree], level = 1)
        new_operands = []
        for operand in tree.operands:
            if operand is not None:
                new_operands.append(operand.insert_intersection())
        tree.operands = new_operands
        return tree

    def lossless_transform(self):
        """
        This routine applies lossless transformations to the tree.
        """
        tree = deepcopy(self)
        tree = tree.fix_useless_nodes()
        tree = tree.fix_empty_region()
        return tree

    def lossy_transform(self):
        """
        This routine transforms any prepared tree to a canonical tree.
        """
        tree = deepcopy(self)
        for i, a_tree in enumerate(tree.operands):
            complete = a_tree.all_terminal()
            while not all(complete):
                if len(np.array(a_tree.operands)[complete].tolist()) == 0:
                    new_operands = []
                else:
                     new_operands = np.array(a_tree.operands)[complete].tolist()
                for operand in np.array(a_tree.operands)[~complete]:
                    new_operands.append(operand.to_terminal())
                tree.operands[i].operands = new_operands
                complete = a_tree.all_terminal()
        return tree
        
    def fix_useless_nodes(self):
        """
        This routine removes any redundant node from any prepared tree.
        """
        tree = deepcopy(self)
        if tree.operator == '||':
            for i, operand in enumerate(tree.operands):
                if operand.operator == '&':
                    tree.operands[i] = operand.clear_useless()
                else:
                    print "The tree is not prepared (2nd level)"
        else:
            print "The tree is not prepared (1st level)"
        return tree

    def clear_useless(self):
        """
        Transforms any tree by removing all redundant nodes
        """
        complete = self.useless_clear_complete() and self.redund_clear_complete()
        while not complete:
            for i, operand in enumerate(self.operands):
                if isinstance(operand, Tree):
                    if len(operand.operands) < 2:
                        self.operands.remove(operand)
                        self.operands += operand.operands
                    elif operand.operator == self.operator:
                        self.operands.remove(operand)
                        self.operands += operand.operands
                    else:
                        self.operands[i] = operand.clear_useless()
            complete = self.useless_clear_complete() and self.redund_clear_complete()
        return self    

    def useless_clear_complete(self):
        """
        Checks if a tree has any useless nodes. 
        If yes, it returns False, else True.
        """
        flag = True
        for operand in self.operands:
            if isinstance(operand, Tree):
                if len(operand.operands) < 2:
                    return False
                else:
                    flag = operand.useless_clear_complete()
        return flag    

    def redund_clear_complete(self):
        """
        Checks if a tree has any redundant nodes. 
        If yes, it returns False, else True.
        """
        flag = True
        for operand in self.operands:
            if isinstance(operand, Tree):
                if self.operator == operand.operator:
                    return False
                else:
                    flag = operand.redund_clear_complete()
        return flag

    def fix_empty_region(self):
        tree = deepcopy(self)
        if tree.operator == '||':
            for i, operand in enumerate(tree.operands):
                if operand.operator == '&':
                    tree.operands[i] = operand.clear_empty()
                else:
                    print "The tree is not prepared (2nd level)"
        else:
            print "The tree is not prepared (1st level)" 
        return tree

    def clear_empty(self):
        tree = deepcopy(self)
        complete = tree.empty_clear_complete()
        while not complete:
            for i, subtree in enumerate(tree.operands):
                if isinstance(subtree, Empty):
                    if tree.operator == '||':
                        if all(map(isinstance,tree.operands,
                            [Empty]*len(tree.operands))):
                            tree = Empty()
                        else:    
                            tree.operands.remove(subtree)
                    elif tree.operator == '&' and tree.level != 2:
                        tree = Empty()
                        break
                    elif tree.operator == '&' and tree.level == 2:
                        tree.operands = [Empty()]
                elif isinstance(subtree, Tree):
                    tree.operands[i] = subtree.clear_empty()
            complete = tree.empty_clear_complete()
        return tree
  
    def empty_clear_complete(self):
        flag = True
        for operand in self.operands:
            if isinstance(operand, Empty):
                if self.level == 2 and len(self.operands) == 1:
                    pass
                else:
                    return False
            elif isinstance(operand, Tree):
                flag = operand.empty_clear_complete()
        return flag            

    def all_terminal(self):
        terminal = np.array(map(isinstance,self.operands,
            [Terminal]*len(self.operands)))
        return terminal    

    def to_terminal(self):
        tree = deepcopy(self)
        terminal = tree.all_terminal()
        while not all(terminal):
            for i,operand in enumerate(self.operands):
                tree.operands[i] = operand.to_terminal()
            terminal = tree.all_terminal()
        return tree.over_approx()  

    def over_approx(self):
        tree = deepcopy(self)
        if tree.operator == '||':
            return self.over_approx_or()
        elif tree.operator == '&':
            return self.over_approx_and()
        else:
            print "unidentified operator {}".format(tree.operator)    

    def over_approx_or(self):
        try:
            from cvxpy import (Variable, Semidef, Variable, Minimize, Problem, 
            log_det, bmat)
            tree = deepcopy(self)
            tau = Variable(len(tree.operands))
            X = Semidef(tree.operands[0].P.shape[0])
            b_ = Variable(tree.operands[0].P.shape[0])
            constraints = []
            for i in range(len(tree.operands)):   
                X11 = X - tau[i]*tree.operands[i].A
                X12 = b_ - tau[i]*tree.operands[i].b
                X13 = np.zeros((tree.operands[0].P.shape[0],tree.operands[0].P.shape[0]))
                X21 = (b_ - tau[i]*tree.operands[i].b).T
                X22 = -1 - tau[i]*tree.operands[i].c
                X23 = b_.T
                X31 = np.zeros((tree.operands[0].P.shape[0],tree.operands[0].P.shape[0]))
                X32 = b_
                X33 = -X
                constraints.append((bmat([[X11, X12, X13], [X21, X22, X23],
                 [X31, X32, X33]]) << 0))
                constraints.append(tau[i] >= 0)    
            objective = Minimize(-log_det(X))
            prob = Problem(objective, constraints)
            result = prob.solve(solver='CVXOPT')
            A = np.array(X.value)
            b = np.array(b_.value)
            c = np.dot(b.T,b) - 1
            return Ellipsoid(A=A,b=b,c=c,approx=True)
        except:
            "Make sure it's an ellipse!"
            return Empty()

    def over_approx_and(self):
        return Empty(approx=True)            
            
class CanonTree(Tree):
    def __init__(self, tree, approx=False):
        self.operator = tree.operator
        self.operands = tree.operands
        self.approx = approx    
    
class HalfSpace(Terminal):
    def __init__(self, point, vector, approx=False):
        Terminal.__init__(self, approx)
        self.point = np.array(point)
        self.shape = self.point.shape
        self.vector = np.transpose(np.array(vector))
        self.b = np.dot(self.vector, np.transpose(self.point))[0][0]

    @property
    def point(self):
        """I'm the 'coord' property."""
        return self._point

    @point.setter
    def point(self, value):
        if is_point(value):
            self._point = np.transpose(value)
        elif is_point(np.transpose(value)):
            self._point = value
        else:
            print value
            print "The input must be a point, the given is {}".format(value)
            sys.exit(1)

    @point.deleter
    def point(self):
        del self._point  

    @property
    def vector(self):
        """I'm the 'coord' property."""
        return self._vector

    @vector.setter
    def vector(self, value):
        if is_point(value):
            if value.shape[0] == self.point.shape[0] and value.shape[1] == self.point.shape[1]:
                self._vector = value
            elif value.shape[0] == self.point.shape[1] and value.shape[1] == self.point.shape[0] and value.shape[0]==1:
                self._vector = value
            else:
                print "The vector's dimensions {} doesn't agree with the"\
                " point's dimensions {}".format(value.shape, self.point.shape)
                sys.exit(1)
        else:
            print "The input must be a vector, the given is {}".format(value)
            sys.exit(1)

    @vector.deleter
    def vector(self):
        del self._vector        
        
    def __str__(self, depth=None):
        return 'HalfSpace ({}, {})'.format(self.point.tolist(), 
            self.vector.tolist()) 
    
    def __repr__(self):
        return 'HalfSpace ({}, {})'.format(self.point, self.vector)          

    def flatten(self):
        return 'HalfSpace ({}, {})'.format(self.point, self.vector)     
    
class Empty(Terminal):
    def __init__(self, approx=False):
        Terminal.__init__(self, approx)
        self.name = 'Empty'

    def __str__(self, depth=None):
        return "Empty()"                 

    def __repr__(self):
        return "{}".format(self.name)  

    def flatten(self):
        return "{}".format(self.name)  

class Workspace(Terminal):
    def __init__(self, approx=False):
        Terminal.__init__(self, approx)
        self.name = 'Workspace'

    def __str__(self, depth=None):
        return "Workspace()"                 

    def __repr__(self):
        return "{}".format(self.name)  

    def flatten(self):
        return "{}()".format(self.name)  
       
class Ellipsoid(Terminal):
    def __init__(self, P=None, xc=None, A=None, b=None, c=None, approx=False):
        Terminal.__init__(self, approx)
        if A is None or b is None or c is None:
            self.P = PDMatrix(P)
            self.xc = CenterPoint(xc)
            self.A = P
            self.b = np.dot(np.array(P),np.array(xc))
            self.c = np.dot(np.dot(np.array(xc).transpose(),P),np.array(xc)) - 1
        else:
            self.P = P
            self.xc = xc
            self.A = A
            self.b = b
            self.c = c
      
    @property
    def P(self):
        """
        I'm the 'P' property.
        """
        return self._P
    
    @P.setter
    def P(self, value):
        if is_PD(value) or value==None:
            self._P = value
        else:
            print "The matrix P is not a PD matrix!"
            sys.exit(1)

    @P.deleter
    def P(self):
        del self._P   
        
    @property
    def xc(self):
        """
        I'm the 'c' property.
        """
        return self._xc
    
    @xc.setter
    def xc(self, value):
        if is_center_point(value):
            if dim_check(self.P, value):
                self._xc = value
            else:
                print "The dimensions are not compatible. P is {} while xc is"\
                " {}. ".format(self.P.shape,value.shape[0])
                sys.exit(1)
        elif value==None:
            self._xc = value
        else:
            print "The point xc is not a center point!"
            sys.exit(1)

    @xc.deleter
    def xc(self):
        del self._xc          

    def __str__(self, depth=None):
        return 'Ellipsoid ({}, {})'.format(self.P.matrix.tolist(), 
            self.xc.coord.tolist())     

    def __repr__(self):
        if self.P is not None:
            return 'Ellipsoid ({}, {})'.format(self.P.matrix, self.xc.coord)
        else:
            return 'Ellipsoid (A:{}, b:{}, c:{})'.format(self.A, self.b, self.c)

    def flatten(self):
        return 'Ellipsoid ({}, {})'.format(self.P.matrix, self.xc.coord)

    def plot(self,color='r'):
        import matplotlib.pyplot as plt
        n = 100000
        lis = np.zeros((n,2))
        for i in range(0,n,4):
            pos1 = random() * 5
            neg1 = random() * -5
            pos2 = random() * 5
            neg2 = random() * -5
            lis[i] = np.array([pos1,pos2])
            lis[i+1] = np.array([neg1,pos2])
            lis[i+2] = np.array([pos1,neg1])
            lis[i+3] = np.array([neg1,neg2])
        y1 = np.array([x for x in lis if (-.1 <= np.dot(np.dot(x.T,self.A),x) 
            + 2*np.dot(self.b.T,x) + self.c <= 0)])
        plt.scatter(y1[:,1],y1[:,0],color=color)
        
    
    def show(self):
        import matplotlib.pyplot as plt
        plt.show()
        
# def plot_multiple(ellipse_list):
#     for ellipse in ellipse_list:
#         ellipse.plot()
#     ellipse_list[0].show()
        
# def main():
#     c = [1,2]
#     A = [[1,0],[0,1]]
#     E = Ellipsoid(A,c)
#     h = HalfSpace([1,2],[4,5])
#     th = Empty()

# if __name__ == '__main__':
#     main()
"""
.. module:: cvx_gen
   :platform: Unix
   :synopsis: 

.. moduleauthor:: Mohammed Alshiekh


"""

from region import *

def tree_leaf_count(tree):
    """
    This function takes a tree and returns its number of leaf.
    """
    n = 0
    for operand in tree.operands:
        if isinstance(operand,Terminal):
            n += 1
        if isinstance(operand,Tree):
            n += tree_leaf_count(operand)
    return n

def total_leaf_count(nodes,edges):
    """
    This function takes a tagged automaton and returns the total number of leafs in all of its trees.
    """
    n = 0
    for key, node in nodes.iteritems():
        n += tree_leaf_count(node['region'])
    for key, edge in edges.iteritems():
        n += tree_leaf_count(edge['region'])
    return n

def _pre_cvx(nodes,edges,constants):
    string = '%dymanics\n'
    string += 'A = '+Matrix(constants['A']).mmat()+';\n'
    string += 'B = '+Matrix(constants['B']).mmat()+';\n'
    string += 'n = size(A,1);\n'
    string += 'm = size(B,2);\n'
    string += 'Q = eye(n);\n'.format()
    string += 'R = eye(m);\n'.format()
    # the dimensions of the final state (origin) must be equal to the workspace's dimensions
    string += 'xf = [0;0];\n'.format()
    # The initial position should be set by the user
    string += 'x0 = [1;1];\n'.format()
    # state transition cost should be set by the user
    string += '%state transition cost\n'
    string += 's = 1;\n'
    string += '%number of states\n'
    string += 'Nq = {};\n'.format(len(nodes.keys()))         
    return string

def _cvx_prog(nodes,edges):

    def s_procedure(op):
        if isinstance(op,HalfSpace):
            return "\t\t- tau({i})*[zeros(n), zeros(n,m), 0.5*{c}' ; zeros(m,n), zeros(m), 0; 0.5*{c}, zeros(1,m), -{b}]...\n".format(i=c,b=op.b,c=Matrix(op.point).mmat())
        elif isinstance(op,Ellipsoid):
            return "\t\t- tau({i})*[{P}, zeros(n,m), {r}' ; zeros(m,n), zeros(m), 0; {r}, zeros(1,m), {c}]...\n".format(i=c,P=Matrix(op.A).mmat(),r=Matrix(op.b).mmat(),c=op.c)  
        else:
            print 'This region is unknown: {}. The resulting cvx prog will probably be incorrect. Please modify the automaton.'.format(op)
            return ''
    
    c = 1     
    
    # starting the sdp program and declaring the variables
    string = 'cvx_begin sdp;\n'
    string += '\t variable P(n,n,Nq);\n'
    string += '\t variable r(n,1,Nq);\n'
    string += '\t variable t(1,Nq);\n'
    string += '\t variable tau({},1);\n'.format(total_leaf_count(nodes,edges))
    
    # maximizing the objective function
    string += "\t maximize( quad_form(x0,P(:,:,1)) + 2*r(:,:,1)'*x0 + t(:,1) );\n"
    
    # subject to:
    string += '\t for i=1:Nq\n'
    string += '\t\t P(:,:,i) == semidefinite(n);\n'
    string += '\t end\n'
    string += '\t tau(:) >= 0;\n'
    
    # state constraints
    for key, node in nodes.iteritems():
        string += '%{} \n'.format(key)
        tree = node['region']
        for operand in tree.operands:
            string += '\n'
            if isinstance(operand,Tree):
                string += "\t 0 <= [A'*P(:,:,{i}) + P(:,:,{i})'*A + Q, P(:,:,{i})*B,  A'*r(:,:,{i}); ...\n".format(i=int(key[2:])+1)
                string += "\t       B'*P(:,:,{i})',                  R,           B'*r(:,:,{i}); ...\n".format(i=int(key[2:])+1)
                string += "\t       r(:,:,{i})'*A,                   r(:,:,{i})'*B, 0  ]...\n".format(i=int(key[2:])+1)
                for op in operand.operands:
                    string += s_procedure(op)
                    c += 1
                string = string[:-4]
                string += ';\n'
                
	# transition constraints
    for key, edge in edges.iteritems():
        string += '\n'
        tree = edge['region']
        for operand in tree.operands:
            string += '%{} \n'.format(key)
            if isinstance(operand,Tree):
                string += "\t 0 <= [P(:,:,{i}),    r(:,:,{i}),   zeros(n,1); ...\n".format(i=int(key[1][2:])+1)
                string += "\t       r(:,:,{i})',   zeros(m),   zeros(m,1); ...\n".format(i=int(key[1][2:])+1)
                string += "\t       zeros(1,n),  zeros(1,m), t(:,{i}) + s] ...\n".format(i=int(key[1][2:])+1)
                string += "\t     -[P(:,:,{i}),    r(:,:,{i}),   zeros(n,1); ...\n".format(i=int(key[0][2:])+1)
                string += "\t       r(:,:,{i})',   zeros(m),   zeros(m,1); ...\n".format(i=int(key[0][2:])+1)
                string += "\t       zeros(1,n),  zeros(1,m), t(:,{i}) + s] ...\n".format(i=int(key[0][2:])+1)
                for op in operand.operands:
                    string += s_procedure(op)
                    c += 1
                string = string[:-4]
                string += ';\n'
                
    # accepting state contraints        
    for key, node in nodes.iteritems():
        if node['accepting']:
            string += '% final \n'  
            string += "\tquad_form(xf,P(:,:,{i})) + 2*r(:,:,{i})'*xf + t(:,{i}) == 0;\n".format(i=int(key[2:])+1)
    string += 'cvx_end;'
    return string
      
def to_cvx(nodes,edges,constants):
    """
    This function assumes that all trees are in canonical form.
    """
    string = _pre_cvx(nodes,edges,constants)
    string += _cvx_prog(nodes,edges)
    return string
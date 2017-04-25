from region import *
from Automata import RegularLanguage
from sys import exit

def tag_nodes(graph, i=1):
    """takes in a graph and returns a dict of nodes
    """
    nodes = dict()
    history = dict()
    state_info_map = dict()
    for source, transitions in graph.iteritems():
        state_info_map[source] = list(transitions)[0][2:]
        for transition in list(transitions):
            if transition[1] not in nodes.keys():
                history[transition[1]] = [transition[0]]
            elif transition[0] not in history[transition[1]]:
                history[transition[1]].append(transition[0])
        for transition in list(transitions):
            nodes[transition[1]] = Tree('||', history[transition[1]])
    for key, value in nodes.iteritems():
        nodes[key] = {'obj':Objective('p_{}'.format(key[2:]),
                      'q_{}'.format(key[2:]),'r_{}'.format(key[2:])),
                      'region': value,
                      'accepting': state_info_map[key][0], 
                      'initial': state_info_map[key][1]}
    return nodes

def tag_edges(graph):
    """
    takes in a graph and returns a dict of edges
    """
    edges = dict()
    for source, transitions in graph.iteritems():
        for transition in list(transitions):
            edges[(source,transition[1])] = {'region': Tree('||', 
                                            [transition[0]])}
    return edges

def convert(mapping,spec,tupl):
    return (tupl[0], mapping[spec][tupl[1]],tupl[2],tupl[3])

class SymbolTable(object):
    """
    This class creates an instance of a symbol table.

    Args:
        static (bool): 

    Attributes:
        container (dict):
        static (bool): 
    """    
    def __init__(self, static = True):
        self.container = dict()
        self.static = static
    
    def insert(self, var_name, scope, value):
        """
        """
        if scope in self.container.keys():
            if var_name in self.container[scope].keys() and self.static:
                print "Variable {} is already defined".format(var_name)
                exit(1)
            else:
                if scope in ['ap','letter','system','spec']:
                    self.container[scope][var_name] = eval(value)
                elif scope == 'region':
                    self.container[scope][var_name] = value
                elif scope == 'aut':
                    if var_name == 'accepting' or var_name == 'initial':
                        self.container[scope][var_name] = map(int,value)
                    elif var_name == 'edges':
                        self.container[scope][var_name] = map(eval,value)
                    elif var_name == 'n_nodes':
                        self.container[scope][var_name] = eval(value)
                    else:
                        self.container[scope][var_name] = value
                elif value[0] == '[':
                    self.container[scope][var_name] = '('+value+')'
                elif value[0] != '(' or value[-1] != ')':
                    self.container[scope][var_name] = value
                else:
                    self.container[scope][var_name] = value  
        else:
            self.container[scope] = dict()
            if scope in ['ap','letter','system', 'spec']:
                self.container[scope][var_name] = eval(value)
            elif scope == 'region':
                self.container[scope][var_name] = eval(value)
            elif scope == 'aut':
                if var_name == 'accepting' or var_name == 'initial':
                    self.container[scope][var_name] = map(int,value)
                elif var_name == 'edges':
                    self.container[scope][var_name] = map(eval,value)
                elif var_name == 'n_nodes':
                    self.container[scope][var_name] = eval(value)
                else:
                    self.container[scope][var_name] = value
                
            else:
                self.container[scope][var_name] = value
    
    def lookup(self, var_name, scope, neg=False):
        """
        """
        try:
            if neg:
                value = str((-1*np.array(eval(self.container[scope][var_name]))).tolist())
                return value
            else:
                value = self.container[scope][var_name]
                return value
        except KeyError:
            print 'The scope {} or variable {} is undefined'.format(scope, 
                var_name)
            exit(1)
            
    def rename_states(self):
        """
        This routine goes over all the specifications and changes the 
        state names from frozensets to q_i.
        """
        mapping = {}
        keys = {}
        for spec, graph in self.container['spec'].iteritems():
            mapping[spec] = dict()
            keys[spec] = list(self.container['spec'][spec].graph.iteritems())
            i = 0
            for state, graph in keys[spec]:
                mapping[spec][state] = 'q_{}'.format(i)       
                self.container['spec'][spec].graph[mapping[spec][state]] = self.container['spec'][spec].graph[state]
                del self.container['spec'][spec].graph[state]
                i += 1
        for spec, graph in self.container['spec'].iteritems():
            i = 0
            for state, graph in keys[spec]: 
                length = len(list(self.container['spec'][spec].graph[mapping[spec][state]]))
                self.container['spec'][spec].graph[mapping[spec][state]] =\
                set(map(convert,[mapping]*length,[spec]*length,
                    list(self.container['spec'][spec].graph[mapping[spec][state]])))
                i += 1
        return self
                
    def get_tagged_nodes(symbol_table):
        """
        """
        if 'aut' not in symbol_table.container.keys():
            nodes = {}
            for spec in symbol_table.container['spec'].keys():
                nodes[spec] =  tag_nodes(symbol_table.container['spec'][spec].graph)
            for spec in symbol_table.container['spec'].keys():        
                for key in nodes[spec].keys():
                    nodes[spec][key]['region'] = nodes[spec][key]['region'].to_canon_tree()  
        else:
            new_keys = ['q_'+str(s) for s in range(symbol_table.container['aut']['n_nodes'])]
            nodes = dict.fromkeys(new_keys)
            for node in nodes.keys():
                nodes[node] = {}
                if isinstance(symbol_table.container['aut'][node],str):
                    nodes[node]['region'] = eval(symbol_table.container['aut'][node]).to_canon_tree()
                else:
                    nodes[node]['region'] = symbol_table.container['aut'][node].to_canon_tree()
                if int(node[2:]) in symbol_table.container['aut']['accepting']:
                    nodes[node]['accepting'] = True
                else:
                    nodes[node]['accepting'] = False
                    
                if int(node[2:]) in symbol_table.container['aut']['initial']:
                    nodes[node]['initial'] = True
                else:
                    nodes[node]['initial'] = False                    
                    
        return nodes

    def get_tagged_edges(symbol_table):
        """
        """
        if 'aut' not in symbol_table.container.keys():
            edges = {}
            for spec in symbol_table.container['spec'].keys():
                edges[spec] =  tag_edges(symbol_table.container['spec'][spec].graph)
            for spec in symbol_table.container['spec'].keys():        
                for key in edges[spec].keys():
                    edges[spec][key]['region'] = edges[spec][key]['region'].to_canon_tree() 
        else:
            new_keys = [s for s in symbol_table.container['aut'].keys() if isinstance(s,tuple)]
            edges = dict.fromkeys(new_keys)
            for key in new_keys:
                edges[key] = {}
                if isinstance(symbol_table.container['aut'][key],str):
                     edges[key]['region'] = eval(symbol_table.container['aut'][key]).to_canon_tree() 
                else:
                    edges[key]['region'] = symbol_table.container['aut'][key].to_canon_tree() 
        return edges
        

from collections import defaultdict
import inspect
class Graph(object):
    """ Graph data structure, directed by default. """

    def __init__(self, connections = None, directed=True):
        self.graph = defaultdict(set)
        self._directed = directed
        # if connections is not None:
        #     self.add_connections(connections)

    # def add_connections(self, connections):
    #     """ Add connections (list of tuple pairs) to graph """

    #     for node1, node2 in connections:
    #         self.add(node1, node2)

    def add(self, node1, letter, node2, accepting=False,initial=False):
        """ Add connection between node1 and node2 """
        # curframe = inspect.currentframe()
        # calframe = inspect.getouterframes(curframe, 2)
        # print 'caller name:', calframe[1][3]
        self.graph[node1].add((letter,node2,accepting,initial))
        #self.graph[node1].add((1,node2,terminal))

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self.graph.iteritems():
            try:
                cxns.remove(node)
            except KeyError:
                pass
        try:
            del self._graph[node]
        except KeyError:
            pass

    # def is_connected(self, node1, node2):
    #     """ Is node1 directly connected to node2 """

    #     return node1 in self.graph and node2 in self.graph[node1]
    
    # def __str__(self):
    #     return '{}({})'.format(self.__class__.__name__, dict(self.graph))
    
    # def __repr__(self):
    #     return '{}({})'.format(self.__class__.__name__, dict(self.graph))
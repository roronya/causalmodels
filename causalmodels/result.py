import os
from graphviz import Digraph
from causalmodels.interface import ResultInterface


class Result(ResultInterface):
    def __init__(self, order, matrix, labels):
        self.order = order
        self.matrix = matrix
        self.labels = labels

    def draw(self, output_name='result', format='png'):
        graph = Digraph(format=format, engine='dot')
        graph.attr('node', shape='circle')
        for label in self.labels:
            graph.node(str(label), str(label))
        for i in self.order:
            for j in self.order:
                if self.matrix[i][j] != 0:
                    graph.edge(str(self.labels[j]),
                               str(self.labels[i]),
                               str(self.matrix[i][j]))
        graph.render(output_name)
        return graph

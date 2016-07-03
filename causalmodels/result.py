import os
from graphviz import Digraph
from causalmodels.interface import ResultInterface


class Result(ResultInterface):
    def __init__(self, order, matrix, sorted_matrix, sorted_data, sorted_labels):
        self.order = order
        self.matrix = matrix
        self.sorted_matrix = sorted_matrix
        self.sorted_data = sorted_data
        self.sorted_labels = sorted_labels

    def plot(self, output_name='result', format='png'):
        graph = Digraph(format=format, engine='dot')
        graph.attr('node', shape='circle')
        for label in self.sorted_labels:
            graph.node(str(label), str(label))
        for i in self.order:
            for j in self.order:
                if self.sorted_matrix[i][j] != 0:
                    graph.edge(str(self.sorted_labels[j]),
                               str(self.sorted_labels[i]),
                               str(self.sorted_matrix[i][j]))
        graph.render(output_name, cleanup=True)
        return graph

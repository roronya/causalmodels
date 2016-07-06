import os
import numpy as np
from graphviz import Digraph
from causalmodels.interface import ResultInterface


class Result(ResultInterface):
    def __init__(self, order, matrix, sorted_matrix, sorted_data, sorted_labels):
        self.order = order
        self.matrix = matrix
        self.sorted_matrix = sorted_matrix
        self.sorted_data = sorted_data
        self.sorted_labels = sorted_labels

    def plot(self, output_name='result', format='png', threshold=0):
        B = self.sorted_matrix.copy()
        if threshold:
            for i, B_i in enumerate(B):
                for j, B_i_j in enumerate(B_i):
                    if np.abs(B_i_j) < threshold:
                        B[i][j] = 0
        graph = Digraph(format=format, engine='dot')
        graph.attr('node', shape='circle')
        for label in self.sorted_labels:
            graph.node(str(label), str(label))
        for i in self.order:
            for j in self.order:
                if B[i][j] != 0:
                    graph.edge(str(self.sorted_labels[j]),
                               str(self.sorted_labels[i]),
                               str(B[i][j]))
        graph.render(output_name, cleanup=True)
        return graph

import os
import numpy as np
from graphviz import Digraph
from causalmodels.interface import ResultInterface


class Result(ResultInterface):
    def __init__(self, order, matrix, data, labels):
        self.order = order
        self.matrix = matrix
        self.data = data
        self.labels = labels

    def plot(self, output_name='result', format='png', threshold=0):
        B = self.matrix[:, self.order]
        if threshold:
            for i, B_i in enumerate(B):
                for j, B_i_j in enumerate(B_i):
                    if np.abs(B_i_j) < threshold:
                        B[i][j] = 0
        graph = Digraph(format=format, engine='dot')
        for label in self.labels:
            graph.node(str(label), str(label))
        sorted_labels = self.labels[self.order]
        for i in self.order:
            for j in self.order:
                if B[i][j] != 0:
                    graph.edge(str(sorted_labels[j]),
                               str(sorted_labels[i]),
                               str(B[i][j]))
        graph.render(output_name, cleanup=True)
        return graph

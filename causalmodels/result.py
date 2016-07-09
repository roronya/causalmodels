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

    def plot(self, output_name="result", format="png", threshold=0):
        graph = Digraph(format=format, engine="dot")
        for label in self.labels:
            graph.node(label)
        for i, m_i in enumerate(self.matrix):
            for j, m_i_j in enumerate(m_i):
                if m_i_j != 0 and np.abs(m_i_j) >= threshold:
                    graph.edge(self.labels[j],
                               self.labels[i],
                               str(m_i_j))
        graph.render(output_name, cleanup=True)
        return graph

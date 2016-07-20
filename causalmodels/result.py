import os
import numpy as np
import random
from graphviz import Digraph
from causalmodels.interface import ResultInterface


class Result(ResultInterface):
    def __init__(self, order, permuted_matrix, data, labels):
        self.order = order
        self.permuted_matrix = permuted_matrix
        self.data = data
        self.labels = labels

    @property
    def permuted_matrix(self):
        return self._permuted_matrix

    @property
    def matrix(self):
        return self._matrix

    @permuted_matrix.setter
    def permuted_matrix(self, permuted_matrix):
        self._permuted_matrix = permuted_matrix
        P = np.eye(len(self.order))[self.order]
        self._matrix = np.dot(np.dot(P.T, permuted_matrix), P)

    @property
    def data(self):
        return self._data

    @property
    def permuted_data(self):
        return self._permuted_data

    @data.setter
    def data(self, data):
        self._data = data
        P = np.eye(len(self.order))[self.order]
        self._permuted_data = np.dot(data, P.T)

    @property
    def labels(self):
        return self._labels

    @property
    def permuted_labels(self):
        return self._permuted_labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        self._permuted_labels = labels[self.order]

    def plot(self, output_name="result", format="png", threshold=0.01, decimal=3):
        graph = Digraph(format=format)
        graph.attr("graph", layout="dot", splines="true", overlap="false")
        graph.attr("node", shape="circle")
        for label in self.labels:
            graph.node(label)
        for i, m_i in enumerate(self.matrix):
            for j, m_i_j in enumerate(m_i):
                if np.abs(round(m_i_j, 3)) >= threshold:
                    graph.edge(self.labels[j],
                               self.labels[i],
                               str(round(m_i_j, decimal)))
        graph.render(output_name, cleanup=True)
        return graph

class ConvolutionResult(ResultInterface):
    def __init__(self, instantaneous_order, permuted_instantaneous_matrix, permuted_convolution_matrixes, data, labels):
        self.instantaneous_order = instantaneous_order
        self.matrixes = (permuted_instantaneous_matrix, permuted_convolution_matrixes)
        self.data = data
        self.labels = labels

    @property
    def permuted_matrixes(self):
        return self._permuted_matrixes

    @property
    def matrixes(self):
        return self._matrixes

    @matrixes.setter
    def matrixes(self, m):
        permuted_instantaneous_matrix = m[0]
        permuted_convolution_matrixes = m[1]
        P = np.eye(len(self.instantaneous_order))[self.instantaneous_order]
        permuted_matrixes = np.empty((
                                permuted_convolution_matrixes.shape[0] + 1,
                                permuted_instantaneous_matrix.shape[0],
                                permuted_instantaneous_matrix.shape[1]))
        for i, matrix in enumerate(permuted_matrixes):
            if i == 0:
                permuted_matrixes[i] = permuted_instantaneous_matrix
            else:
                permuted_matrixes[i] = permuted_convolution_matrixes[i - 1]
        matrixes = np.empty((
                        permuted_convolution_matrixes.shape[0] + 1,
                        permuted_instantaneous_matrix.shape[0],
                        permuted_instantaneous_matrix.shape[1]))
        for i, matrix in enumerate(matrixes):
            if i == 0:
                matrixes[i] = np.dot(np.dot(P.T, permuted_instantaneous_matrix), P)
            else:
                matrixes[i] = np.dot(np.dot(P.T, permuted_convolution_matrixes[i - 1]), P)
        self._permuted_matrixes = permuted_matrixes
        self._matrixes = matrixes

    @property
    def data(self):
        return self._data

    @property
    def permuted_data(self):
        return self._permuted_data

    @data.setter
    def data(self, data):
        self._data = data
        P = np.eye(len(self.instantaneous_order))[self.instantaneous_order]
        self._permuted_data = np.dot(data, P.T)

    @property
    def labels(self):
        return self._labels

    @property
    def permuted_labels(self):
        return self._permuted_labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels
        self._permuted_labels = labels[self.instantaneous_order]

    def plot(self, output_name="result", format="png", separate=False, decimal=3, threshold=0.01, integration=False):
        if integration:
            integration_matrix = self.matrixes.sum(axis=0)
            graph = Digraph(format=format)
            graph.attr("graph", layout="dot", splines="true", overlap="false")
            graph.attr("node", shape="circle")
            for label in self.labels:
                graph.node(label)
            for i, m_i in enumerate(integration_matrix):
                for j, m_i_j in enumerate(m_i):
                    if np.abs(round(m_i_j, decimal)) >= threshold:
                        graph.edge(self.labels[j],
                                   self.labels[i],
                                   str(round(m_i_j, decimal)))
            graph.render(output_name, cleanup=True)
            return graph
        else:
            def generate_random_color():
                return "#{:X}{:X}{:X}".format(*[random.randint(0, 255) for i in range(3)])
            graph = Digraph(format=format)
            graph.attr("graph", layout="dot", splines="true", overlap="false")
            graph.attr("node", shape="circle")
            legend = Digraph("cluster_legend")
            legend.attr("graph", rankdir="LR")
            legend.attr("node", style="invis")
            lags = ["t"] + ["t_{}".format(i) for i in range(1, len(self.matrixes))]
            for label in self.labels:
                graph.node(label)
            for lag, matrix in zip(lags, self.matrixes):
                color = generate_random_color()
                legend.edge("s_{}".format(lag),
                            "d_{}".format(lag),
                            lag,
                            color=color)
                for i, m_i in enumerate(matrix):
                    for j, m_i_j in enumerate(m_i):
                        if round(np.abs(m_i_j), decimal) >= threshold:
                            graph.edge(self.labels[j],
                                       self.labels[i],
                                       str(round(m_i_j, decimal)),
                                       color=color)
            graph.subgraph(legend)
            graph.render(output_name, cleanup=True)
            return graph

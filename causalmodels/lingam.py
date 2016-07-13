import numpy as np
import scipy as sp
import scipy.linalg
import sklearn.linear_model as lm
import statsmodels.tsa.api as tsa
from graphviz import Digraph
from tqdm import tqdm, trange
from causalmodels.interface import ModelInterface
from causalmodels.result import Result
from causalmodels.exception import *
from causalmodels.interface import ResultInterface

def residual(X_j, X_i):
    return X_i - (np.cov(X_i, X_j, bias=1)[0][1] / np.var(X_j)) * X_j

def gram_matrix(X, sigma):
    return np.exp([[(-1.0 / (2.0 * (sigma**2))) * (X_i - X_j) ** 2 for X_j in X] for X_i in X])

def centering_gram_matrix(X):
    n = X.shape[0]
    P = np.array([[-1 / n if i != j else (n - 2) / n for j in range(n)] for i in range(n)])
    return P.dot(X).dot(P)

def log_determinant(X):
    U = sp.linalg.lu(X, permute_l=True)[1]
    U = np.diag(U)
    U = np.abs(U)  # マイナスの値の log を計算するのを防ぐ
    U = np.log(U).sum()
    return U

def MIkernel(y_1, y_2, kappa=0.02, sigma=0.5):
    K_1 = centering_gram_matrix(gram_matrix(y_1, sigma))
    K_2 = centering_gram_matrix(gram_matrix(y_2, sigma))
    # K_k を作る
    n = K_1.shape[0]
    K_k_left_upper = (lambda x: np.dot(x, x))(K_1 + n * kappa / 2 * np.eye(n))
    K_k_right_upper = np.dot(K_1, K_2)
    K_k_left_lower = np.dot(K_2, K_1)
    K_k_right_lower = (lambda x: np.dot(x, x))(K_2 + n * kappa / 2 * np.eye(n))
    K_k = np.concatenate([np.concatenate([K_k_left_upper, K_k_right_upper], axis=1),
                          np.concatenate([K_k_left_lower, K_k_right_lower], axis=1)])
    D_k_left_upper = K_k_left_upper.copy()
    D_k_right_upper = np.zeros((n, n))
    D_k_left_lower = np.zeros((n, n))
    D_k_right_lower = K_k_right_lower.copy()
    D_k = np.concatenate([np.concatenate([D_k_left_upper, D_k_right_upper], axis=1),
                          np.concatenate([D_k_left_lower, D_k_right_lower], axis=1)])
    log_K_k = log_determinant(K_k)
    log_D_k = log_determinant(D_k)
    return (-1 / 2) * (log_K_k - log_D_k)

def Tkernel(X, j, U):
    Tkernel = np.sum([MIkernel(X[:, j], residual(X[:, j], X[:, i])) for i in U])
    return Tkernel

class DirectLiNGAM(ModelInterface):
    def __init__(self, data, labels=None):
        self.data = np.array(data)
        self.labels = np.array(labels) if labels is not None else np.array([str(i) for i in range(data.shape[1])])

    def estimate_coefficient(self, regression, alpha=0.1, max_iter=1000):
        X = self.data[:, self.order].copy()
        n = X.shape[1]
        coef = np.zeros((n, n))
        for i, X_i in reversed(list(enumerate(X.T))):
            if i == 0:
                break
            A = X[:, :i]
            b = X[:, i]
            if regression == "lasso":
                model = lm.Lasso(alpha=alpha, max_iter=max_iter)
            elif regression == "ridge":
                model = lm.Ridge(alpha=alpha)
            else:
                model = lm.LinearRegression()
            model.fit(A, b)
            c = model.coef_
            for j, X_j in enumerate(X.T):
                if j == i:
                    break
                coef[i][j] = c[j]
        return coef

    def fit(self, regression="LinearRegression", alpha=0.1, max_iter=1000):
        X = self.data.copy()
        K = []
        for i in trange(X.shape[1], desc="calcurating 1st independence"):
            U = [k for k, v in enumerate(X.T) if k not in K]
            X_m_index = sorted([(Tkernel(X, j, U), j) for j in tqdm(U, desc="calcurating Tkernel value")])[0][1]
            for k in U:
                if k != X_m_index:
                    X[:, k] = residual(X[:, k], X[:, X_m_index])
            K.append(X_m_index)
        # data を K 順に並び替える
        self.order = K
        B = self.estimate_coefficient(regression=regression, alpha=alpha, max_iter=max_iter)
        # 元の順に戻す
        P = np.eye(len(K))[K]
        B = np.dot(np.dot(P.T, B), P)
        self.result = Result(order=K,
                             matrix=B,
                             data=self.data,
                             labels=self.labels)
        return self.result

class SVARDirectLiNGAM(DirectLiNGAM):
    def __init__(self, data, labels=None):
        super(SVARDirectLiNGAM, self).__init__(data, labels)
        self.var_model = tsa.VAR(data)

    def select_order(self, maxlag=None, verbose=True):
        return self.var_model.select_order(maxlags=maxlag, verbose=verbose)

    def fit_var(self, data, maxlags=15, ic="aic"):
        var_result = self.var_model.fit(maxlags=maxlags, ic=ic)
        return var_result

    def fit(self, regression="LinearRegression", alpha=0.1, max_iter=1000, maxlags=15, ic="aic"):
        var_result = self.fit_var(self.data, maxlags=maxlags, ic=ic)
        lag_order = var_result.k_ar
        data = self.data[lag_order:] - var_result.forecast(self.data[0:lag_order], self.data.shape[0]-lag_order)
        super_result = super(SVARDirectLiNGAM, self).fit(regression=regression, alpha=alpha, max_iter=max_iter)
        B_0 = super_result.matrix
        matrixes = np.empty((lag_order+1, B_0.shape[0], B_0.shape[1]))
        var_coefficient = var_result.coefs
        for i, m_i in enumerate(matrixes):
            if i == 0:
                matrixes[i] = B_0
            else:
                matrixes[i] = np.linalg.solve(np.eye(B_0.shape[0]) - B_0, var_coefficient[i-1])
        self.result = SVARDirectLiNGAMResult(instantaneous_order=super_result.order,
                                             matrixes=matrixes,
                                             data=self.data,
                                             labels=self.labels)
        return self.result

class SVARDirectLiNGAMResult(ResultInterface):
    def __init__(self, instantaneous_order, matrixes, data, labels):
        self.instantaneous_order = instantaneous_order
        self.matrixes = matrixes
        self.data = data
        self.labels = labels

    def plot(self, output_name="result", format="png", separate=False, threshold=0):
        if separate:
            instantaneous_matrix = self.matrixes[0]
            instantaneous_graph = Digraph("cluster_t", format=format)
            instantaneous_graph.attr("node", shape="circle")
            for src, dst in zip(self.labels[self.instantaneous_order][:-1], self.labels[self.instantaneous_order][1:]):
                instantaneous_graph.edge("{label}(t)".format(label=src),
                                         "{label}(t)".format(label=dst),
                                         style="invis")
            for i, m_i in enumerate(instantaneous_matrix):
                for j, m_i_j in enumerate(m_i):
                    if np.abs(m_i_j) > threshold:
                        instantaneous_graph.edge("{label}(t)".format(label=self.labels[j]),
                                                 "{label}(t)".format(label=self.labels[i]),
                                                 str(round(m_i_j, 3)))
            instantaneous_graph.render("{output_name}(t)".format(output_name=output_name), cleanup=True)
            tau = self.matrixes.shape[0]
            lags = ["t_{0}".format(t) for t in range(1, tau)]
            layers = [Digraph("cluster_{lag}".format(lag=lag)) for lag in lags]
            graphs = [Digraph(format=format) for lag in lags]
            for lag, matrix, layer, graph in zip(lags, self.matrixes[1:], layers, graphs):
                layer.attr("node", shape="circle")
                for label in self.labels:
                    layer.node("{label}({lag})".format(label=label, lag=lag))
                graph.subgraph(instantaneous_graph)
                graph.subgraph(layer)
                graph.attr("graph", layout="dot", splines="true", overlap="false")
                for src, dst in zip(self.labels[self.instantaneous_order][:-1], self.labels[self.instantaneous_order][1:]):
                    graph.edge("{label}({lag})".format(label=src, lag=lag),
                               "{label}({lag})".format(label=dst, lag=lag),
                               style="invis")
                for i, m_i in enumerate(matrix):
                    for j, m_i_j in enumerate(m_i):
                        if np.abs(m_i_j) > threshold:
                            graph.edge("{label}({lag})".format(label=self.labels[j], lag=lag),
                                       "{label}({lag})".format(label=self.labels[i], lag="t"),
                                        str(round(m_i_j, 3)))
                graph.render("{output_name}({lag})".format(output_name=output_name, lag=lag), cleanup=True)
            return graphs
        else:
            graph = Digraph(format=format)
            graph.attr("graph", layout="dot", splines="true", overlap="false")
            graph.attr("node", shape="circle")
            tau = self.matrixes.shape[0]
            lags = ["t"] + ["t_{0}".format(t) for t in range(1, tau)]
            layers = [Digraph("cluster_{0}".format(lag)) for lag in lags]
            for lag, layer in zip(lags, layers):
                layer.attr("graph", label=lag)
                for label in self.labels:
                    layer.node("{label}({lag})".format(label=label, lag=lag))
            for layer in layers:
                graph.subgraph(layer)
            for lag, matrix in zip(lags, self.matrixes):
                if lag != "t":
                    for src, dst in zip(self.labels[self.instantaneous_order][:-1], self.labels[self.instantaneous_order][1:]):
                        graph.edge("{label}({lag})".format(label=src, lag=lag),
                                   "{label}({lag})".format(label=dst, lag=lag),
                                   style="invis")
                for i, m_i in enumerate(matrix):
                    for j, m_i_j in enumerate(m_i):
                        if np.abs(m_i_j) > threshold:
                            if lag == "t":
                                graph.edge("{label}({lag})".format(label=self.labels[j], lag=lag),
                                           "{label}({lag})".format(label=self.labels[i], lag=lag),
                                           str(round(m_i_j, 3)))
                            else:
                                graph.edge("{label}({lag})".format(label=self.labels[j], lag=lag),
                                           "{label}({lag})".format(label=self.labels[i], lag="t"),
                                           str(round(m_i_j, 3)))
            graph.render(output_name, cleanup=True)
            return graph

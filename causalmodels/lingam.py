import numpy as np
import scipy as sp
import scipy.linalg
import sklearn.linear_model as lm
import statsmodels.tsa.api as tsa
from tqdm import tqdm, trange
from causalmodels.interface import ModelInterface
from causalmodels.result import Result
from causalmodels.exception import *

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

    def __init__(self):
        self.order = None
        self.matrix = None

    def estimate_coefficient(self, X, regression, alpha=0.1, max_iter=1000):
        n = X.shape[1]
        B = np.zeros((n,n))
        for i, X_i in reversed(list(enumerate(X.T))):
            if i == 0:
                break
            A = X[:, :i]
            b = X[:, i]
            if regression == 'lasso':
                model = lm.Lasso(alpha=alpha, max_iter=max_iter)
            elif regression == 'ridge':
                model = lm.Ridge(alpha=alpha)
            else:
                model = lm.LinearRegression()
            model.fit(A, b)
            c = model.coef_
            for j, X_j in enumerate(X.T):
                if j == i:
                    break
                B[i][j] = c[j]
        return B

    def fit(self, data, labels=None, regression='LinearRegression', alpha=0.1, max_iter=1000):
        X = data.copy()
        K = []
        for i in trange(X.shape[1], desc='calcurating 1st independence'):
            U = [k for k, v in enumerate(X.T) if k not in K]
            X_m_index = sorted([(Tkernel(X, j, U), j) for j in tqdm(U, desc='calcurating Tkernel value')])[0][1]
            for i in U:
                if i != X_m_index:
                    X[:, i] = residual(X[:, i], X[:, X_m_index])
            K.append(X_m_index)
        # data を K 順に並び替える
        X = data[:, K]
        B = self.estimate_coefficient(X, regression=regression, alpha=alpha, max_iter=max_iter)
        # 元の順に戻す
        self.matrix = np.zeros(B.shape)
        for i, k in enumerate(K):
            self.matrix[k] = B[i]
        self.order = K
        self.sorted_matrix = B
        self.sorted_data = X
        self.sorted_labels = labels[K] if labels is not None else None
        return self.predict()

    def predict(self):
        if self.matrix is None:
            raise NotYetFitError()
        return Result(order=self.order, matrix=self.matrix, sorted_matrix=self.sorted_matrix, sorted_data=self.sorted_data, sorted_labels=self.sorted_labels)

class SVARDirectLiNGAM(DirectLiNGAM):
    def fit_var(self, data, maxlags=15, ic='aic'):
        var_model = tsa.VAR(data)
        var_estimation = var_model.fit(maxlags=maxlags, ic=ic)
        return var_estimation

    def fit(self, data, labels=None, regression='LinearRegression', alpha=0.1, max_iter=1000, maxlags=15, ic='aic'):
        var_estimation = self.fit_var(data, maxlags=maxlags, ic=ic)
        lag_order = var_estimation.k_ar
        data = data[lag_order:] - var_estimation.forecast(data[0:lag_order], data.shape[0]-lag_order)
        result = super(SVARDirectLiNGAM, self).fit(data, labels=labels, regression=regression, alpha=alpha, max_iter=max_iter)
        result.var_estimation = var_estimation
        return result

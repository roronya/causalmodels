import numpy as np
import scipy as sp
from causalmodels.interface import ModelInterface
from causalmodels.result import Result
from causalmodels.exception import *

class DirectLiNGAM(ModelInterface):
    def __init__(self):
        self.order = None
        self.matrix = None

    """
    implement of DirectLiNGAM
    """
    def fit(self, data, labels=None):
        #<!-- inner function
        def T_kernel(x_j, X, K):
            return np.sum([MI_kernel(x_j, x_i - (np.cov(x_j, x_i, bias=1)[1][0] / np.var(x_j)) * x_j)
                           for i, x_i in enumerate(X)
                           if i not in K])

        def MI_kernel(y_1, y_2, sigma=0.5, kappa=0.02):
            # グラム行列の作成
            def gram_matrix(X):
                return np.exp([[-1.0 / (2.0 * (sigma**2)) * (x_i - x_j)**2 for x_j in X] for x_i in X])

            K_1 = gram_matrix(y_1)
            K_2 = gram_matrix(y_2)

            # グラム行列のセンタリング
            def centering_gram_matrix(X):
                n = X.shape[0]
                P = np.array([[-1/n if i !=j else (n-2)/n for j in range(n)] for i in range(n)])
                return P.dot(X).dot(P)

            K_1 = centering_gram_matrix(K_1)
            K_2 = centering_gram_matrix(K_2)

            # K_k の作成
            n = K_1.shape[0]
            K_k_left_upper = (lambda x: np.dot(x, x))(K_1 + n*kappa/2*np.eye(n))
            K_k_right_upper = np.dot(K_1, K_2)
            K_k_left_lower = K_k_right_upper.copy()
            K_k_right_lower = (lambda x: np.dot(x, x))(K_2 + n*kappa/2*np.eye(n))
            K_k = np.concatenate([np.concatenate([K_k_left_upper, K_k_right_upper], axis=1),
                                np.concatenate([K_k_left_lower, K_k_right_lower], axis=1)])

            # D_k の作成
            D_k_left_upper = K_k_left_upper.copy()
            D_k_right_upper = np.zeros((n, n))
            D_k_left_lower = np.zeros((n, n))
            D_k_right_lower = K_k_right_lower.copy()
            D_k = np.concatenate([np.concatenate([D_k_left_upper, D_k_right_upper], axis=1),
                                np.concatenate([D_k_left_lower, D_k_right_lower], axis=1)])

            # np.linalg.det では nan になってしまうので行列式の計算時から log をとる
            def log_determinant(X):
                U = sp.linalg.lu(X, permute_l=True)[1]
                U = np.diag(U)
                U = np.abs(U) # マイナスの値の log を計算するのを防ぐ
                U = np.log(U).sum()
                return U

            log_K_k = log_determinant(K_k)
            log_D_k = log_determinant(D_k)
            return (-1/2) * (log_K_k - log_D_k)
        # inner function --!>

        X = data.copy()
        K = []
        n = data.shape[0]
        B = np.zeros((n, n))
        for k, x_k in enumerate(X):
            # 最も独立な変数を見つける
            x_m_index = np.argmin([T_kernel(x_j, X, K) if j not in K else float('inf')
                                           for j, x_j in enumerate(X)])
            K.append(x_m_index)

            # 最も独立な変数から残りの変数への影響を計算する
            x_m = X[x_m_index]
            for l , x_l in enumerate(X):
                if l not in K:
                    B[l][x_m_index] = np.cov(x_l, x_m, bias=1)[0][1] / np.var(x_m)

            # 残りの変数から最も独立だった変数の影響を取り除く 
            for l, x_l in enumerate(X):
                if l not in K:
                    X[l] = x_l - (np.cov(x_m, x_l, bias=1)[0][1] / np.var(x_m)) * x_m

        self.order = K
        self.matrix = B
        self.labels = labels if labels is not None else range(X.shape[0])

        return self.predict()

    def predict(self):
        if self.matrix is None:
            raise NotYetFitError()
        return Result(order=self.order, matrix=self.matrix, labels=self.labels)

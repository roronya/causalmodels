import functools
import multiprocessing as mp
import numpy as np
import scipy as sp
from scipy import linalg
import causalmodels.base.model as base


class SparseDirectLiNGAM(base.Model):
    """
    implement of DirectLiNGAM.
    """

    def __init__(self):
        self.processes = 1
        self.threshold = 0.2

    def fit(self, data):
        """
        因果関係を推論する
        Args:
        X: 観測データ
        prior_matrix: 事前知識行列
        Returns:
        K: 因果関係の順番の一次元配列
        B: 因果関係を表す下三角行列
        Example:
        K:[0,2,1]
        B:[[0,0,0],
        [1,0,0],
        [2,3,0]]
        だったら、入力された X は 0 => 2 => 1 という因果関係があって、その関係は
        行列 B のようになっているということ
        """
        def generate_gaussian_kernel(sigma=0.5):
            """ガウシアンカーネル生成する
            引数の分散を元にガウシアンカーネル関数を生成する
            カーネル関数は引数に値がペアで格納されたタプルを受け取りカーネル値を算出し返す
            sigma は論文では 0.5 だったのでデフォルトで 0.5 としている
            Args:
                sigma: ガウシアン分布の分散
            Returns:
                ガウシアンカーネル関数
            """
            return lambda x_i, x_j: np.exp((-1 / (2 * (sigma**2))) * ((x_i - x_j)**2))

        def generate_gram_matrix(x, kernel_function=generate_gaussian_kernel()):
            """グラム行列を生成する
            Args:
                x: 一次元配列
                kernel_function: カーネル関数
            Returns:
                x のサイズ n とすると、 n x n の グラム行列
            """
            return np.array([[kernel_function(xi, xj) for xj in x] for xi in x])

        def calc_log_determinant(X):
            """行列式の対数を計算する
            行列式を計算する途中で log を取る
            行列式がとても大きい値になった場合に後から log を取って誤差が生じるのを対策している
            そもそも行列式が Inf になるから log とって総和取る
            Args:
                X: 行列
            Returns:
                行列式の対数
            """
            U = sp.linalg.lu(X, permute_l=True)[1]
            U = np.diag(U)
            U = np.abs(U)  # マイナスの値の log を計算するのを防ぐ
            U = np.log(U).sum()
            return U

        def centering_gram_matrix(K):
            """グラム行列をセンタリングする
            Args:
                K: グラム行列
            Returns:
                センタリングしたグラム行列
            """
            n = len(K)
            P = np.array(
                [[-1 / n if i != j else (n - 2) / n for j in range(n)] for i in range(n)])
            return P.dot(K).dot(P)

        def calc_MI_kernel_value(x, r, kappa=2 * 10**(-2)):
            """カーネル法を利用して相互情報量を計算する
            Args:
                x: 観測データ
                r: 残差
                kappa: 小さい正の値 論文では 2*10^(-2) だった
            Returns:
                相互情報量
            """
            K_1 = generate_gram_matrix(x)
            K_2 = generate_gram_matrix(r)
            K_1 = centering_gram_matrix(K_1)
            K_2 = centering_gram_matrix(K_2)
            n = len(x)
            numer_11 = (K_1 + ((n * kappa) / 2) * np.eye(n, n))
            numer_11 = numer_11.dot(numer_11)
            numer_12 = K_1.dot(K_2)
            numer_21 = K_2.dot(K_1)
            numer_22 = (K_2 + ((n * kappa) / 2) * np.eye(n, n))
            numer_22 = numer_22.dot(numer_22)
            numer = np.r_[np.c_[numer_11, numer_12], np.c_[numer_21, numer_22]]

            denom_11 = numer_11
            denom_12 = np.zeros((n, n))
            denom_21 = np.zeros((n, n))
            denom_22 = numer_22
            denom = np.r_[np.c_[denom_11, denom_12], np.c_[denom_21, denom_22]]
            log_numer = calc_log_determinant(numer)
            log_denom = calc_log_determinant(denom)
            return (-1 / 2) * (log_numer - log_denom)

        def calc_residual(x_i, x_j):
            """残差を計算する
            x_i が x_j から受けている影響を取り除く
            Args:
                x_i: 一次元配列 これからx_j の成分を取り除きたい
                x_j: 一次元配列 この成分を x_i から取り除きたい
            Returns:
                x_i から x_j の成分を取り除いた一次元配列
            """
            return x_i - (calc_causal_effect(x_i, x_j) * x_j)

        def calc_causal_effect(x_i, x_j):
            """x_j から x_i へどれくらい影響を及ぼすのか計算する
            Args:
                x_i: 一次元配列
                x_j: 一次元配列
            Returns:
                x_j が x_i へ与える影響値
            """
            return np.cov(x_i, x_j, bias=1)[0][1] / np.var(x_j)

        def calc_T_kernel_value(X, j):
            """j 以外の 添字 i について X[i] から j の要素を取り除いた残差の相互情報量をの総和を計算する
            R_j は R_j[i] で X[i] から X[j] の成分を取り除いた残差を得られる
            残差と観測データの相互情報量が最も小さいものが一番独立
            Args:
                X: 観測データ
                j: 現在注目している観測データの添字
            Returns:
                X[j] の独立度を測る指標
            """
            R_j = np.array([calc_residual(X[i], X[j])
                            for i in X.keys() if i != j])
            T_kernel_value = np.array(
                [calc_MI_kernel_value(X[j], R_j_i) for R_j_i in R_j]).sum()
            return T_kernel_value

        def find_most_independent_variable(X, processes):
            """最も独立な変数を見つける
            観測データのそれぞれにおいて、独立度を calc_T_kernel_value() で計算し
            最も小さな値を持つ変数の添字を返す
            Args:
                X: 観測データ
            Returns:
                最も独立な変数の添字
            """
            partial_calc_T_kernel_value = functools.partial(
                calc_T_kernel_value, X)  # pool.map() は引数が 1 つの関数しか使えないので partial() で部分適用をする
            if processes > 1:
                pool = mp.Pool(processes=processes)
                T = np.array(list(zip(pool.map(partial_calc_T_kernel_value,
                                               X.keys()), X.keys())), dtype=[('x', float), ('y', int)])
            else:
                T = np.array(list(zip(map(partial_calc_T_kernel_value, X.keys()), X.keys())), dtype=[
                            ('x', float), ('y', int)])
            min = (float('inf'), 0)
            for t in T:
                if t[0] < min[0]:
                    min = t
            return min[1]

        def find_indirect_causal_effect(causal_effect_matrix):
            adjust_matrix = causal_effect_matrix.astype(bool)
            indirect_causal_effect = np.zeros(causal_effect_matrix.shape).astype(bool)
            A_n = adjust_matrix
            for n in range(len(adjust_matrix) - 1):
                A_n = A_n.dot(adjust_matrix)
                indirect_causal_effect = np.logical_or(indirect_causal_effect, A_n)
            return indirect_causal_effect

        n = data.shape[0]
        B = np.zeros((n, n))
        order = []
        X = {i: data[i] for i in range(n)}
        X_org = X.copy()

        while (len(X) > 0):
            k = find_most_independent_variable(X, self.processes)

            order .append(k)
            for i in X.keys():
                if i != k:
                    # k => i への因果効果
                    B[i][k] = calc_causal_effect(X[i], X[k])
            X = {i: calc_residual(X[i], X[k]) for i in X.keys() if i != k}

        # B を 下三角行列に並び替える
        inference = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if j < i:
                    inference[i][j] = B[order[i]][order[j]]

        # エッジを枝刈りする
        for i in range(n):
            for j in range(n):
                if inference[i][j] < self.threshold:
                    inference[i][j] = 0
        indirect_causal_effect = find_indirect_causal_effect(inference)

        # indirect_causal_effect が true のところを見つけて
        for i in range(n):
            for j in range(n):
                if indirect_causal_effect[i][j]:
                    R = X_org[order[i]]
                    for k in range(n):
                        if k == j or k == i:
                            continue
                        if inference[i][k] > 0 and not indirect_causal_effect[i][k]:
                            R = calc_residual(R, X_org[order[k]])
                    inference[i][j] = calc_causal_effect(R, X_org[order[j]])
                    if inference[i][j] < self.threshold:
                        inference[i][j] = 0

        results = base.Results(causal_order=order,
                               causal_inference_matrix=inference)

        return results

    def predict(self):
        pass

    def set_processes(self, n):
        self.processes = n
        return self

    def set_threshold(self, n):
        self.threshold = n
        return self

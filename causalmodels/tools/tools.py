import numpy as np
import scipy as sp
from scipy import linalg


def generate_gaussian_kernel(sigma=0.5):
    """ガウシアンカーネル生成する
    引数の分散を元にガウシアンカーネル関数を生成する
    カーネル関数は引数に値がペアで格納されたタプルを受け取りカーネル値を算出し返す
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

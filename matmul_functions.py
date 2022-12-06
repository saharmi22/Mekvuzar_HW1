import math

import numpy as np
from numba import njit, cuda
import timeit

from typing import Tuple


def matmul_transpose_trivial(X):
    XXt = np.zeros((X.shape[0], X.shape[0]))
    for row in range(X.shape[0]):
        for column in range(row + 1):
            sum_mul = 0
            for index in range(X.shape[1]):
                sum_mul += X[row][index] * X[column][index]
            XXt[row][column] = sum_mul
            if row != column:
                XXt[column][row] = sum_mul
    return XXt
    # raise NotImplementedError("To be implemented")


@njit
def matmul_transpose_numba(X):
    XXt = np.zeros((X.shape[0], X.shape[0]))
    for row in range(X.shape[0]):
        for column in range(row + 1):
            sum_mul = 0
            for index in range(X.shape[1]):
                sum_mul += X[row][index] * X[column][index]
            XXt[row][column] = sum_mul
            if row != column:
                XXt[column][row] = sum_mul
    return XXt
    # raise NotImplementedError("To be implemented")


def matmul_transpose_gpu(X):
    threads_per_block = 1024
    blocks_per_grid = 1
    XXt = np.zeros((X.shape[0], X.shape[0]))
    gpu_XXt = cuda.to_device(XXt)
    gpu_X = cuda.to_device(X)
    # Now start the kernel
    matmul_kernel[blocks_per_grid, threads_per_block](gpu_X, gpu_XXt)
    XXt = gpu_XXt.copy_to_host()
    return XXt


@cuda.jit
def matmul_kernel(A, C):
    thread_num = cuda.threadIdx.x
    r = thread_num // A.shape[0]
    c = thread_num % A.shape[0]
    while r < A.shape[0] and c < A.shape[0]:
        if r >= c:
            tot_sum = 0
            for k in range(A.shape[1]):
                tot_sum += A[r, k] * A[c, k]
            C[c, r] = tot_sum
            if r != c:
                C[r, c] = tot_sum
        c += 1024
        while c > A.shape[0]:
            c -= A.shape[0]
            r += 1


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) # we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    matmul_comparison()

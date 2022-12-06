import numpy as np
from numba import cuda, njit, prange, float32
import timeit
import os
# enabling CUDA simulation:
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'


def max_cpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    return np.array([[max(A[i][j], B[i][j]) for i in range(A.shape[0])] for j in range(A.shape[1])])



@njit(parallel=True)
def max_numba(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    C = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j] = max(A[i, j], B[i, j])
    return C

@cuda.jit
def max_kernel(A, B):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    if block_id < A.shape[0] and thread_id < A.shape[1]:
        cuda.atomic.max(A[block_id], thread_id, B[block_id, thread_id])


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    threads_per_block = 1000
    blocks_per_grid = 1000
    gpu_A = cuda.to_device(A)
    gpu_B = cuda.to_device(B)
    max_kernel[blocks_per_grid, threads_per_block](gpu_A, gpu_B)
    C = gpu_A.copy_to_host()
    return C
# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    print('     [*] CPU:', timer(max_cpu))
    print('     [*] Numba:', timer(max_numba))
    print('     [*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    max_comparison()
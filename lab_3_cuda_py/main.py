from __future__ import division
import numpy as np
import time
from numba import cuda
import math

def serial(A, D):
    for i in range(n):
        for k in range(n):
            sum = 0
            for j in range(m):
                sum += pow((A[i][j] - A[k][j]), 2)
            D[i][k] = sum
    return D

@cuda.jit
def cudaGlobal(A, D):
    row, col = cuda.grid(2)
    if row < D.shape[0] and col < D.shape[1]:
        sum = 0
        for j in range(m):
            sum += pow((A[row, j] - A[col, j]), 2)
        D[row, col] = sum

@cuda.jit
def cudaShared(A, D):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=np.int16)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= D.shape[0] and y >= D.shape[1]:
        return
    sum = 0
    for i in range(int(n / TPB)):
        sA[tx, ty] = A[x, ty + i * TPB]
        cuda.syncthreads()
        for j in range(TPB):
            sum += pow((A[tx, j] - A[ty, j]), 2)
        cuda.syncthreads()
    D[x, y] = sum

if __name__ == "__main__":
    m = 256
    n = 512
    TPB = 16

    A = np.random.rand(n, m)

    D = np.zeros(n * n, dtype=np.int16)
    D.shape = (n, n)

    aForGlobal = cuda.to_device(A)
    dForGlobal = cuda.device_array((n, n))
    threadsPerBlockGlobal = (TPB, TPB)
    blocksPerGridXGlobal = int(math.ceil(n / threadsPerBlockGlobal[0]))
    blocksPerGridGlobal = (blocksPerGridXGlobal)

    aForShared = cuda.to_device(A)
    dForShared = cuda.device_array((n, n))
    threadsPerBlockShared = (TPB, TPB)
    blocksPerGridXShared = int(math.ceil(n / threadsPerBlockShared[1]))
    blocksPerGridShared = (blocksPerGridXShared)

    serialTime = time.time()
    serial(A, D)
    print(time.time() - serialTime)

    globalTime = time.time()
    cudaGlobal[blocksPerGridGlobal, threadsPerBlockGlobal](aForGlobal, dForGlobal)
    print(time.time() - globalTime)

    sharedTime = time.time()
    cudaShared[blocksPerGridShared, threadsPerBlockShared](aForShared, dForShared)
    print(time.time() - sharedTime)


from cuda_methods import *
import numpy as np
import time

"""Matrix Addition"""

rows = 20000
cols = 20000

a = np.random.normal(0, 1,(rows, cols)).astype('float32')
b = np.random.normal(0, 1,(rows, cols)).astype('float32')
c = np.zeros((rows, cols)).astype('float32')


start_time = time.time()
cuda_sum(a, b, c, rows, cols)

print(f'cuda matrix addition: with time {time.time() - start_time} sec:\n', c)

start_time = time.time()
c = a + b

print(f'python matrix addition: with time {time.time() - start_time} sec:\n', c)


"""Matrix Multiplication"""

a_rows = 10000
a_cols = 10000

b_rows = 10000
b_cols = 10000

a = np.random.normal(0, 1,(a_rows, a_cols)).astype('float32')
b = np.random.normal(0, 1,(b_rows, b_cols)).astype('float32')
c = np.zeros((a_rows, b_cols)).astype('float32')


start_time = time.time()
cuda_mul(a, b, c,  a_rows, a_cols, b_rows, b_cols)

print(f'cuda matrix multiplication with time {time.time() - start_time} sec:\n', c)


start_time = time.time()
cuda_blas_mul(a, b, c,  a_rows, a_cols, b_rows, b_cols)

print(f'cuda blas matrix multiplication with time {time.time() - start_time} sec:\n', c)


start_time = time.time()
c = np.dot(a, b)

print(f'npy matrix multiplication with time {time.time() - start_time} sec:\n', c)


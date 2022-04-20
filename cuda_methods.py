from ctypes import *
import ctypes
import os


#for cublas methods (my pc doesn`t see this path in environment variables)
os.add_dll_directory('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin')

def __get_cuda_addition():
    dll = ctypes.CDLL('./dependencies/mat_add.dll', mode = ctypes.RTLD_GLOBAL)
    func = dll.cudaMatAdd
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_size_t]
    return func


def __get_cuda_multiplication():
    dll = ctypes.CDLL('./dependencies/mat_mul.dll', mode = ctypes.RTLD_GLOBAL)
    func = dll.cudaMatMul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_size_t]
    return func


def __get_cuda_blas_multiplication():
    dll = ctypes.CDLL('./dependencies/blas_mat_mul.dll', mode = ctypes.RTLD_GLOBAL)
    func = dll.cudaMatMul
    func.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_size_t]
    return func


__cuda_add = __get_cuda_addition()
__cuda_mul = __get_cuda_multiplication()
__cuda_blas_mul = __get_cuda_blas_multiplication()


def cuda_sum(a, b, c, rows, cols):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))

    __cuda_add(a_p, b_p, c_p, rows, cols)

def cuda_mul(a, b, c, a_rows, a_cols, b_rows, b_cols):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))

    __cuda_mul(a_p, b_p, c_p, a_rows, a_cols, b_rows, b_cols)

def cuda_blas_mul(a, b, c, a_rows, a_cols, b_rows, b_cols):
    a_p = a.ctypes.data_as(POINTER(c_float))
    b_p = b.ctypes.data_as(POINTER(c_float))
    c_p = c.ctypes.data_as(POINTER(c_float))

    __cuda_blas_mul(a_p, b_p, c_p, a_rows, a_cols, b_rows, b_cols)

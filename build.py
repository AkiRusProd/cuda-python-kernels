import os
import subprocess


def create_dll(filename):
    """Create DLL file from CUDA (Windows)"""
    proc = subprocess.Popen(f'nvcc -o dependencies/{filename}.dll  -lcublas -lcurand --shared kernels/{filename}.cu')
    out, err = proc.communicate()

create_dll('mat_mul')
create_dll('mat_add')
create_dll('blas_mat_mul')
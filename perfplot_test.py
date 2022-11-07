

import os


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import itertools
import functools
import numpy as np
import scipy.fft
import mkl_fft
import pyfftw

pyfftw.config.NUM_THREADS = 1
# pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
# pyfftw.config.PLANNER_EFFORT = 'FFTW_PATIENT'
# pyfftw.config.PLANNER_EFFORT = 'FFTW_EXHAUSTIVE'

import perfplot

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1000.)



base_axis = 2


all_base_shape = [
    np.array((1,2,1)),
    np.array((2,2,1)),
    np.array((3,2,1)),
]

n_shapes = len(all_base_shape)



all_real_dtype = [
    # np.float32,
    np.float64,
]
all_complex_dtype = [
    # np.complex64,
    np.complex128,
]
all_dtype_names = [
    # "float32",
    "float64",
]

ntypes = len(all_real_dtype)
assert len(all_real_dtype) == ntypes
assert len(all_complex_dtype) == ntypes
assert len(all_dtype_names) == ntypes


def setup_rfft_vec(N,axis,shape,real_dtype,complex_dtype):
    shape[axis] = N
    vec = np.random.random(shape).astype(dtype=real_dtype)

    return vec
# 
def setup_irfft_vec(N,axis,shape,real_dtype,complex_dtype):
    shape[axis] = N//2+1
    vec = np.random.random(shape).astype(dtype=complex_dtype) + 1j*np.random.random(shape).astype(dtype=complex_dtype)

    return vec




# equality_check = functools.partial(np.allclose, rtol = 1e-3,atol = 1e-5 )
equality_check = None


# 
# vec = setup_rfft_vec(10,2,np.array((1,1,1)),np.float64,np.complex128)
# res = scipy.fft.rfft(vec,axis=2)
# 
# print(vec.shape)
# print(vec.dtype)
# print(res.shape)
# print(res.dtype)
# 
# # 
# vec = setup_irfft_vec(10,2,np.array((1,1,1)),np.float64,np.complex128)
# res = scipy.fft.irfft(vec,axis=2)
# 
# print(vec.shape)
# print(vec.dtype)
# print(res.shape)
# print(res.dtype)
# 
# exit()

all_funs =  [
    [
        functools.partial(np.fft.rfft, axis = base_axis ),
        functools.partial(scipy.fft.rfft, axis = base_axis ),
        functools.partial(mkl_fft.rfft_numpy, axis = base_axis ),
        functools.partial(pyfftw.interfaces.numpy_fft.rfft, axis = base_axis ),
        functools.partial(pyfftw.interfaces.scipy_fft.rfft, axis = base_axis ),
        functools.partial(scipy.fftpack.rfft, axis = base_axis ),
        functools.partial(pyfftw.interfaces.scipy_fftpack.rfft, axis = base_axis ),
    ]  ,  [
        functools.partial(np.fft.irfft, axis = base_axis ),
        functools.partial(scipy.fft.irfft, axis = base_axis ),
        functools.partial(mkl_fft.irfft_numpy, axis = base_axis ),
        functools.partial(pyfftw.interfaces.numpy_fft.irfft, axis = base_axis ),
        functools.partial(pyfftw.interfaces.scipy_fft.irfft, axis = base_axis ),
        # functools.partial(scipy.fftpack.irfft, axis = base_axis ),
        # functools.partial(pyfftw.interfaces.scipy_fftpack.irfft, axis = base_axis ),
    ]  ,
]

backend_names = [
    "Numpy",
    "Scipy",
    "MKL numpy",
    "PyFFTW numpy",
    "PyFFTW scipy",
    "PyFFTW fftpack",
    "Scipy fftpack",
]

all_setups = [
    setup_rfft_vec,
    setup_irfft_vec,
]

all_funs_names = ['RFFT','IRFFT']

n_funs = len(all_funs)


# n_test = 1
n_test = 3

def ranges(*args):
    all_ranges = [range(arg) for arg in args]
    return itertools.product(*all_ranges)

def print_ranges(i_ranges,n_ranges):
    n = len(i_ranges)
    for i in range(n):
        print(f'{i_ranges[i]+1}/{n_ranges[i]}  ',end='')
    
    print("")



for (i_type,i_funs,i_test,i_shape) in ranges(ntypes,n_funs,n_test,n_shapes):

    print_ranges((i_type,i_funs,i_test,i_shape),(ntypes,n_funs,n_test,n_shapes))

    real_dtype = all_real_dtype[i_type]
    complex_dtype = all_complex_dtype[i_type]
    funs = all_funs[i_funs]
    base_shape = all_base_shape[i_shape]

    setup = functools.partial(all_setups[i_funs], real_dtype = real_dtype, complex_dtype = complex_dtype , shape = base_shape, axis = base_axis )

    data_size=[600 * (2 ** k) for k in range(7)]    

    b = perfplot.bench(
        setup = setup,
        kernels = funs,
        labels = backend_names,
        n_range = data_size,
        equality_check = equality_check ,
        target_time_per_measurement = 1.0,
        max_time = 20.,
    )

    filename = f"fft_benchmark_{all_dtype_names[i_type]}_{all_funs_names[i_funs]}_shape{i_shape+1}_test{i_test+1}_relative.png"

    b.save(
        filename ,
        transparent = False,
        relative_to = 2  ,
    )

    filename = f"fft_benchmark_{all_dtype_names[i_type]}_{all_funs_names[i_funs]}_shape{i_shape+1}_test{i_test+1}_absolute.png"

    b.save(
        filename ,
        transparent = False,
    )



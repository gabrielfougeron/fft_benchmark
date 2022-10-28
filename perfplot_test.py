

import os


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


import functools
import numpy as np
import scipy.fft
import mkl_fft
import pyfftw

pyfftw.config.NUM_THREADS = 1
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

import perfplot

pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1000.)

base_shape = np.array((1,1,1))
base_axis = 2


all_dtype = [np.float32,np.float64]
all_dtype_names = ["float32","float64"]


def setup_vec(N,axis,shape,dtype):
    shape[axis] = N
    vec = np.random.random(shape).astype(dtype=dtype)

    pyfftw.interfaces.numpy_fft.rfft(vec,axis=axis)
    pyfftw.interfaces.scipy_fft.rfft(vec,axis=axis)
    pyfftw.interfaces.scipy_fftpack.rfft(vec,axis=axis)

    return vec

def setup_pyfftw(N,axis,shape,dtype,dtype_name,effort):
    pyfftw.config.PLANNER_EFFORT = effort
    shape[axis] = N
    vec = pyfftw.empty_aligned(shape, dtype=dtype_name,n=1)
    fun = pyfftw.builders.rfft(vec)
    vec[...] = np.random.random(shape).astype(dtype=dtype)
    fun()

    return fun

def fftw_rfft(obj):
    
    return obj()


def setup_pyfftw_dropin(N,axis,shape,dtype,dtype_name,effort):
    pyfftw.config.PLANNER_EFFORT = effort
    shape[axis] = N
    vec = pyfftw.empty_aligned(shape, dtype=dtype_name,n=1)
    vec[...] = np.random.random(shape).astype(dtype=dtype)
    pyfftw.interfaces.numpy_fft.rfft(vec,axis=axis)

    return vec


# equality_check = functools.partial(np.allclose, rtol = 1e-3,atol = 1e-5 )
equality_check = None

for i_type in range(len(all_dtype)):

    setup = functools.partial(setup_vec, dtype = all_dtype[i_type], shape = base_shape, axis = base_axis )

    funs = [
        functools.partial(np.fft.rfft, axis = base_axis ),
        functools.partial(scipy.fft.rfft, axis = base_axis ),
        functools.partial(mkl_fft.rfft_numpy, axis = base_axis ),
        functools.partial(pyfftw.interfaces.numpy_fft.rfft, axis = base_axis ),
        functools.partial(pyfftw.interfaces.scipy_fft.rfft, axis = base_axis ),
        functools.partial(pyfftw.interfaces.scipy_fftpack.rfft, axis = base_axis ),

    ]   

    names = [
        "Numpy RFFT",
        "Scipy RFFT",
        "MKL numpy",
        "PyFFTW numpy",
        "PyFFTW scipy",
        "PyFFTW fftpack",
    ]

    data_size=[600 * (2 ** k) for k in range(7)]    

    b = perfplot.bench(
        setup=setup,
        kernels=funs,
        labels = names,
        n_range=data_size,
        equality_check=equality_check ,
        target_time_per_measurement=1.0,
        max_time = 20.,
    )

    b.save(
        f"fft_benchmark_{all_dtype_names[i_type]}.png" ,
        transparent=False,
        relative_to=2  ,
    )





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

# # 
# scipy.fft.register_backend(pyfftw.interfaces.scipy_fft)
# scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)
pyfftw.interfaces.cache.enable()


base_shape = np.array((1,2,1))
base_axis = 2


all_dtype = [np.float32,np.float64]
all_dtype_names = ["float32","float64"]


def setup_vec(N,axis,shape,dtype):
    shape[axis] = N
    vec = np.random.random(shape).astype(dtype=dtype)
    return vec

def setup_pyfftw(N,axis,shape,dtype,dtype_name,effort):
    pyfftw.config.PLANNER_EFFORT = effort
    shape[axis] = N
    vec = pyfftw.empty_aligned(shape, dtype=dtype_name)
    fun = pyfftw.builders.rfft(vec)
    vec[:] = np.random.random(shape).astype(dtype=dtype)
    fun()

    return fun


def fftw_rfft(obj):
    
    return obj()
# equality_check = functools.partial(np.allclose, rtol = 1e-3,atol = 1e-5 )
equality_check = None

for i_type in range(len(all_dtype)):


    # setup = functools.partial(setup_vec, dtype = all_dtype[i_type], shape = base_shape, axis = base_axis )


    setup = [
        functools.partial(setup_vec, dtype = all_dtype[i_type], shape = base_shape, axis = base_axis ),
        functools.partial(setup_vec, dtype = all_dtype[i_type], shape = base_shape, axis = base_axis ),
        functools.partial(setup_vec, dtype = all_dtype[i_type], shape = base_shape, axis = base_axis ),
        functools.partial(setup_pyfftw, dtype = all_dtype[i_type], dtype_name = all_dtype_names[i_type], shape = base_shape, axis = base_axis, effort='FFTW_ESTIMATE' ),
        functools.partial(setup_pyfftw, dtype = all_dtype[i_type], dtype_name = all_dtype_names[i_type], shape = base_shape, axis = base_axis, effort='FFTW_MEASURE' ),
        functools.partial(setup_pyfftw, dtype = all_dtype[i_type], dtype_name = all_dtype_names[i_type], shape = base_shape, axis = base_axis, effort='FFTW_PATIENT' ),
        functools.partial(setup_pyfftw, dtype = all_dtype[i_type], dtype_name = all_dtype_names[i_type], shape = base_shape, axis = base_axis, effort='FFTW_EXHAUSTIVE' ),
    ]

    funs = [
        np.fft.rfft,
        scipy.fft.rfft,
        mkl_fft.rfft_numpy,
        fftw_rfft,
        fftw_rfft,
        fftw_rfft,
        fftw_rfft,
    ]   

    names = [
        "Numpy RFFT",
        "Scipy RFFT",
        "MKL numpy",
        "PyFFTW estimate",
        "PyFFTW measure",
        "PyFFTW patient",
        "PyFFTW exhaustive",
    ]

    data_size=[600 * (2 ** k) for k in range(7)]    

    b = perfplot.bench(
        setup=setup,
        kernels=funs,
        labels = names,
        n_range=data_size,
        equality_check=equality_check ,
        max_time = 10.,
    )

    b.save(
        f"fft_benchmark_{all_dtype_names[i_type]}.png" ,
        transparent=False,
        relative_to=-1  ,
    )





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
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
# pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
# pyfftw.config.PLANNER_EFFORT = 'FFTW_PATIENT'
# pyfftw.config.PLANNER_EFFORT = 'FFTW_EXHAUSTIVE'


pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1000.)


import perfplot
import matplotlib
import matplotlib.pyplot as plt


def ranges(*args):
    all_ranges = [range(arg) for arg in args]
    return itertools.product(*all_ranges)

def print_ranges(i_ranges,n_ranges):
    n = len(i_ranges)
    for i in range(n):
        print(f'{i_ranges[i]+1}/{n_ranges[i]}  ',end='')
    
    print("")


fig_folder = './out'

ext_list = ['png']
# ext_list = ['png','pdf']

if not(os.path.isdir(fig_folder)):

    os.makedirs(fig_folder)

base_axis = 2

all_base_shape = [
    # np.array((1,2,1)),
    np.array((2,2,1)),
    # np.array((3,2,1)),
]

n_shapes = len(all_base_shape)

all_real_dtype = [
    np.float32,
    np.float64,
]
all_complex_dtype = [
    np.complex64,
    np.complex128,
]
all_dtype_names = [
    "float32",
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
n_test = 300

time_per_test = 0.01



data_size = [600 * (2 ** k) for k in range(7)]     
# data_size = [512 * (2 ** k) for k in range(5)]     

n_data = len(data_size)
max_backends = len(all_funs[0])

save_filename = 'all_timings.npy'

all_params_names = [
    all_dtype_names,
    all_funs_names,
    [f'{item}_test' for item in range(n_test)],
    [f'{item[0]}_loops' for item in all_base_shape],
    backend_names,
    [f'{item}_pts' for item in data_size],
]

all_timings_shape = (ntypes,n_funs,n_test,n_shapes,max_backends,n_data)

if os.path.isfile(save_filename):

    print("Benchmark file found. Loading ...")

    all_timings = np.load('all_timings.npy')

    assert (np.array(all_timings.shape) == np.array(all_timings_shape)).all()

else:

    print("Benchmark file not found. Performing benchmark ...")

    print(f"Estimated total time: {time_per_test*ntypes*n_funs*n_test*n_shapes*max_backends*n_data} seconds")

    all_timings = np.zeros(all_timings_shape)

    for (i_type,i_funs,i_test,i_shape) in ranges(ntypes,n_funs,n_test,n_shapes):


        print_ranges((i_type,i_funs,i_test,i_shape),(ntypes,n_funs,n_test,n_shapes))

        real_dtype = all_real_dtype[i_type]
        complex_dtype = all_complex_dtype[i_type]
        funs = all_funs[i_funs]
        base_shape = all_base_shape[i_shape]

        setup = functools.partial(all_setups[i_funs], real_dtype = real_dtype, complex_dtype = complex_dtype , shape = base_shape, axis = base_axis )

        b = perfplot.bench(
            setup = setup,
            kernels = funs,
            labels = backend_names,
            n_range = data_size,
            equality_check = equality_check ,
            target_time_per_measurement = time_per_test,
            max_time = 20.,
        )

        all_timings[i_type,i_funs,i_test,i_shape,0:len(funs),:] = b.timings_s.copy()

    np.save('all_timings.npy',all_timings)

    
color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
n_colors = len(color_list)

linestyle_list = ['solid','dotted','dashed','dashdot']
n_linestyles = len(linestyle_list)

plot_alpha = True
# plot_alpha = False

ax_names = [
    "FP_type"     , # 0,
    "FFT_type"    , # 1
    "Indep_test"  , # 2
    "Array_shape" , # 3
    "FFT_backend" , # 4
    "Data_size"   , # 5   
]

relative_to = None
# relative_to = np.array([2])
n_avg = [2]

n_points = 5

n_lines = [4]

n_plot = []
for i in range(len(all_timings_shape)) :
    if not(i == n_points or i in n_lines or i in n_avg) :
        n_plot.append(i)

plot_ranges = ranges(*[all_timings_shape[i] for i in n_plot])

it = np.zeros((len(ax_names)),dtype=int)

for it_plot in plot_ranges:

    plot_name = ''
    for i in range(len(it_plot)):
        it[n_plot[i]] = it_plot[i]
        plot_name += ' ' +  all_params_names[n_plot[i]][it_plot[i]]

    fig = plt.figure(figsize=(14,10))
    fig.clf()
    ax = fig.add_subplot(1,1,1)

    points_labels = []
    points_ranges = range(all_timings_shape[n_points])
    for it_point in points_ranges:
        points_labels.append(all_params_names[n_points][it_point])

    leg_patch = []

    i_line = -1
    line_ranges = ranges(*[all_timings_shape[i] for i in n_lines])
    for it_line in line_ranges:

        i_line += 1

        line_label = ''
        for i in range(len(it_line)):
            it[n_lines[i]] = it_line[i]
            line_label += ' ' +  all_params_names[n_lines[i]][it_line[i]]
        
        x = []
        y = []
        top_spread_y = []
        bot_spread_y = []

        points_ranges = range(all_timings_shape[n_points])
        for it_point in points_ranges:

            it[n_points] = it_point

            tot_avg = 0

            avg_val = 0.
            min_val = None
            max_val = None

            avg_ranges = ranges(*[all_timings_shape[i] for i in n_avg])
            for it_avg in avg_ranges:

                for i in range(len(it_avg)):
                    it[n_avg[i]] = it_avg[i]
                
                if relative_to is None:
                    rel_val = 1.
                else:
                    it_rel = it.copy()
                    for i in range(len(it_line)):
                        it_rel[n_lines[i]] = relative_to[i]

                    rel_val = all_timings[tuple(it_rel)]

                cur_val = all_timings[tuple(it)]/rel_val

                avg_val += cur_val
                min_val = cur_val if min_val is None else min(cur_val,min_val)
                max_val = cur_val if max_val is None else max(cur_val,max_val)
                tot_avg += 1
            
            avg_val = avg_val / tot_avg

            x.append(it_point)
            y.append(avg_val)
            top_spread_y.append(max_val)
            bot_spread_y.append(min_val)

        # color = color_list[i_line % n_colors]
        color = color_list[it_line[0] % n_colors]

        marker = None

        linestyle = linestyle_list[0]
        # linestyle = linestyle_list[i_line % n_linestyles]
        # linestyle = linestyle_list[it_line[1] % n_linestyles]

        leg_patch.append(matplotlib.patches.Patch(color=color, label=line_label,linestyle=linestyle))

        # plt.plot(x,y,label=line_label,color = color, marker=marker,linestyle=linestyle)
        # plt.text(x[-1], y[-1], line_label,color = color_list[i_line % n_colors])

        # plt.fill_between(x, top_spread_y, bot_spread_y,color=color,alpha=0.2)

        if (plot_alpha):


            x = []
            sorted_vals = []

            points_ranges = range(all_timings_shape[n_points])
            for it_point in points_ranges:

                it[n_points] = it_point
                x.append(it_point)

                it_slice = list(it.copy())
                for i in n_avg:
                    it_slice[i] = slice(None)

                sorted_vals.append(np.sort(all_timings[tuple(it_slice)].reshape(-1)))

                
            sorted_vals = np.array(sorted_vals)

            n_vals = sorted_vals.shape[1]
            n_fills = n_vals // 2
                
            # color = color_list[i_line % n_colors]
            color = color_list[it_line[0] % n_colors]

            alpha = 0.7 / n_fills

            for i_fill in range(n_fills):

                i_bot = i_fill
                i_top = n_vals - 1 - i_fill

                plt.plot(x,sorted_vals[:,i_bot] ,label=line_label,color = color, marker=marker,linestyle=linestyle,alpha=alpha,lw=2.)
                plt.plot(x, sorted_vals[:,i_top],label=line_label,color = color, marker=marker,linestyle=linestyle,alpha=alpha,lw=2.)
                

                # plt.fill_between(x, sorted_vals[:,i_bot], sorted_vals[:,i_top],facecolor=color,alpha=alpha)
                






    # print(points_labels)
    ax.set_xticks(ticks=x,labels=points_labels)
    
    plt.legend(
        handles=leg_patch,    
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
         borderaxespad=0.,
         )
    # fig.tight_layout()
    # ax.set_xscale('log')

    ax.set_yscale('log')
    # ax.set_ylim(bottom=0.)


    for ext in ext_list:

        fig_filename = os.path.join(fig_folder,plot_name+'.'+ext)
        plt.savefig(fig_filename, bbox_inches='tight')

    plt.close('all')



            


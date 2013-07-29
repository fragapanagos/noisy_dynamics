import numpy as np
from matplotlib import pyplot as plt
from pycolorplot.colorgen import ColorGenerator as CG
from distributions import *
import pdb

def dxdt_cubic(x):
    return 8./3 * (-x**3 + x) 

def dxdt_quad_clip(x):
    base = x**2 - 1
    return np.where(np.logical_and(base>0,x>0), 1-x, base)

def local_avg(x, dxdt, dist, dxmin=1, dxmax=1, npts=50):
    """Convolves dxdt with dist around x.

    Uses the region [x-dxmin: x+dxmax]"""
    if dist.sigma == 0:
        return dxdt(x)
    pts, step = np.linspace(x-dxmin, x+dxmax, num=npts, retstep=True)
    return np.sum(dxdt(pts) * dist.pdf(pts-x)) * step

# simulation parameters
T = 4       # total simulation time
dt = 0.001  # simulation timestep
x0s = np.linspace(-1.5,1.5,10)  # initial conditions
x = np.linspace(-1.5,1.5,50)    # space to examine phase

# select dynamical system
# dxdt = dxdt_cubic # cubic system
dxdt = dxdt_quad_clip # quadratic system

# noise parameters 
max_sigma = 1.              # max noise level
sigmas = np.linspace(0,2,9) # noise levels
noise_sources = [gaussian(mu=0, sigma=sigma) for sigma in sigmas] # gaussian
# noise_sources = [uniform(mu=0, sigma=sigma) for sigma in sigmas] # uniform

cgen = CG('red') # color generator for plotting

print 'plotting phase portrait...'
plt.figure('phase')
ax = plt.subplot(111)
ax.plot(x, dxdt(x), color='k')
ax.set_color_cycle(cgen.get_color_list(10))
for x0 in x0s:
    plt.plot(x0, dxdt(x0), 'o', markersize=10)
plt.axhline(color='b')
plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$\dot{x}$', fontsize=18)
plt.savefig('phase')

print 'analyzing phase portrait with noise...'
plt.figure('phase_noise')
for noise_idx, noise_src in enumerate(noise_sources):
    phase_conv = np.zeros(x.shape)
    for i, pt in enumerate(x):
        phase_conv[i] = local_avg(pt, dxdt, noise_src, dxmin=3, dxmax=3)
    plt.subplot(3,3,noise_idx+1)
    plt.plot(x, phase_conv, color='k')
    plt.axhline(color='b')
    locs, labels = plt.xticks()
    plt.xticks(locs, locs, rotation=10)
    # zc_idx = np.where(np.diff(np.sign(phase_conv)))[0][-1]
    # plt.axvline((x[zc_idx]+x[zc_idx+1])/2, color='r')
    plt.title(r'$\sigma=%1.2f$'%sigmas[noise_idx])
plt.subplot(338)
plt.xlabel(r'$x$', fontsize=18)
plt.subplot(334)
plt.ylabel(r'$\dot{x}$', fontsize=18)
plt.tight_layout()
plt.savefig('phase_noise')

print 'running dynamics...'
plt.figure('dynamics')
xtick_loc = [float(i) for i in range(5)]
xtick_lab = [str(float(i)) for i in range(5)]
for noise_idx, noise_src in enumerate(noise_sources):
    ax = plt.subplot(3,3,noise_idx+1)
    ax.set_color_cycle(cgen.get_color_list(10))
    for x0 in x0s:
        now = 0
        states = [x0]
        time = [now]
        state = x0
        while now < T:
            now = now + dt
            state = state + dxdt(state + noise_src.get_sample()) * dt 
            time.append(now)
            states.append(state)
        ax.plot(time, states)
        plt.xticks(xtick_loc, xtick_loc)
    plt.title(r'$\sigma=%1.2f$'%sigmas[noise_idx])
plt.subplot(338)
plt.xlabel(r'$x$', fontsize=18)
plt.subplot(334)
plt.ylabel(r'$\dot{x}$', fontsize=18)
plt.tight_layout()
plt.savefig('dynamics')

print 'done'

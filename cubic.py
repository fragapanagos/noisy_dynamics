import numpy as np
from matplotlib import pyplot as plt
from pycolorplot.colorgen import ColorGenerator as CG
from distributions import *
import pdb

def dxdt(x):
    return 8./3 * (-x**3 + x) 

def conv(x, dxdt, dist, dxmin=10, dxmax=10, npts=50, params=None):
    """Convolves dxdt with dist around x.

    Uses the region [x-dxmin: x+dxmax]"""
    pts, step = np.linspace(x-dxmin, x+dxmax, num=npts, retstep=True)
    return np.sum(dxdt(pts) * dist.f(pts)) * step

T = 4       # total simulation time
dt = 0.001  # timestep
x0s = np.linspace(-1.5,1.5,10)  # initial conditions
sigmas = np.linspace(0,2,9)     # noise levels
noise_sources = [gaussian(mu=0, sigma=sigma) for sigma in sigmas] # noise generators

cgen = CG('red') # color generator for plotting

print 'plotting phase portrait...'
x = np.linspace(-1.5,1.5,50)
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

# print 'analyzing phase portrait with noise...'
# plt.figure('phase_uniform_noise')
# for sigma_idx, sigma in enumerate(sigmas):
#     phase_conv = np.zeros(n)
#     for i in range(len(x)):
#         phase_conv[i] = np.mean(dxdt(np.linspace(x[i]- sigma / 2., x[i] + sigma / 2., 50)))
#     plt.subplot(3,3,sigma_idx+1)
#     plt.plot(x, phase_conv)
#     plt.axhline(color='k')
#     # zc_idx = np.where(np.diff(np.sign(phase_conv)))[0][-1]
#     # plt.axvline((x[zc_idx]+x[zc_idx+1])/2, color='r')
#     plt.title(r'$\sigma=%1.2f$'%sigma)
# plt.subplot(338)
# plt.xlabel(r'$x$', fontsize=18)
# plt.subplot(334)
# plt.ylabel(r'$\dot{x}$', fontsize=18)
# plt.tight_layout()
# plt.savefig('phase_uniform_noise')


print 'running dynamics...'
plt.figure('dynamics')
xtick_loc = [float(i) for i in range(5)]
xtick_lab = [str(float(i)) for i in range(5)]
for sigma_idx, sigma in enumerate(sigmas):
    ax = plt.subplot(3,3,sigma_idx+1)
    ax.set_color_cycle(cgen.get_color_list(10))
    for x0 in x0s:
        now = 0
        states = [x0]
        time = [now]
        state = x0
        while now < T:
            now = now + dt
            state = state + dxdt(state + sigma * np.random.rand() - sigma / 2) * dt # uniform noise
            time.append(now)
            states.append(state)
        ax.plot(time, states)
        plt.xticks(xtick_loc, xtick_loc)
    plt.title(r'$\sigma=%1.2f$'%sigma)
plt.subplot(338)
plt.xlabel(r'$x$', fontsize=18)
plt.subplot(334)
plt.ylabel(r'$\dot{x}$', fontsize=18)
plt.tight_layout()
plt.savefig('dynamics')

print 'done'

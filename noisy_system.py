import numpy as np
from matplotlib import pyplot as plt
from pycolorplot.colorgen import ColorGenerator as CG
from distributions import *
from util import *
import pdb

###  dynamical system functions to choose from ##########
def dxdt_cubic(x, u):
    return 8./3 * (-(x+u)**3 + (x+u)) 

def dxdt_integrator_with_error(x, u):
    return 0.1*np.sin(4*np.pi*x) + u

def dxdt_integrator(x,u):
    return np.zeros(x.shape) + u

def dxdt_exp(x,u):
    return np.exp(-x+u) - 1
#########################################################

### simulation parameters
T = 4       # total simulation time
dt = 0.001  # simulation timestep
x0s = np.linspace(-1.5,1.5,10)  # initial conditions
x = np.linspace(-1.5,1.5,100)   # space to examine phase

### select dynamical system
dxdt = dxdt_exp

### noise parameters 
max_sigma = 5.                      # max noise level
sigmas = np.linspace(0,max_sigma,9) # noise levels
noise_sources = [gaussian(mu=0, sigma=sigma) for sigma in sigmas] # gaussian
# noise_sources = [uniform(mu=0, sigma=sigma) for sigma in sigmas] # uniform

cgen = CG('red') # color generator for plotting

# print 'plotting phase portrait...'
# plt.figure('phase')
# ax = plt.subplot(111)
# ax.plot(x, dxdt(x,0), color='k')
# ax.set_color_cycle(cgen.get_color_list(10))
# for x0 in x0s:
#     plt.plot(x0, dxdt(x0,0), 'o', markersize=10)
# plt.axhline(color='b')
# s_pts = find_stable_pts(x, dxdt(x,0))
# for pt in s_pts:
#     plt.axvline(pt)
# plt.xlabel(r'$x$', fontsize=18)
# plt.ylabel(r'$\dot{x}$', fontsize=18)
# plt.savefig('phase')

# print 'analyzing phase portrait with noise...'
# plt.figure('phase_noise')
# for noise_idx, noise_src in enumerate(noise_sources):
#     phase_conv = np.zeros(x.shape)
#     for i, pt in enumerate(x):
#         phase_conv[i] = local_avg(pt, dxdt, noise_src, dxmin=10, dxmax=10, npts=100)
#     plt.subplot(3,3,noise_idx+1)
#     plt.plot(x, phase_conv, color='k')
#     plt.axhline(color='b')
#     locs, labels = plt.xticks()
#     plt.xticks(locs, locs, rotation=10)
#     s_pts = find_stable_pts(x, phase_conv)
#     for pt in s_pts:
#         plt.axvline(pt)
#     plt.title(r'$\sigma=%1.2f$'%sigmas[noise_idx])
# plt.subplot(338)
# plt.xlabel(r'$x$', fontsize=18)
# plt.subplot(334)
# plt.ylabel(r'$\dot{x}$', fontsize=18)
# plt.tight_layout()
# plt.savefig('phase_noise')

print 'running dynamics...'
plt.figure('dynamics')
xtick_loc = [float(i) for i in range(5)]
xtick_lab = [str(float(i)) for i in range(5)]
for noise_idx, noise_src in enumerate(noise_sources):
    ax = plt.subplot(3,3,noise_idx+1)
    ax.set_color_cycle(cgen.get_color_list(10))
    for x0 in x0s:
        # states = [x0]
        # state = x0
        n_traj = 25
        states = np.zeros((n_traj,4002))
        states[:,0] = x0
        for i in range(n_traj):
            now = 0
            time = [now]
            idx = 0
            while now < T:
                now = now + dt
                idx += 1
                # state = state + dxdt(state, noise_src.get_sample()) * dt # multiplicative noise
                # state = state + dxdt(state, 0) * dt + noise_src.get_sample() * np.sqrt(dt) # additive noise
                states[i,idx] = states[i,idx-1] + dxdt(states[i,idx-1], 0) * dt + noise_src.get_sample() * np.sqrt(dt) # additive noise
                time.append(now)
                # states.append(state)
        # ax.plot(time, states)
        # pdb.set_trace()
        ax.plot(time, np.mean(states,axis=0))
    plt.xticks(xtick_loc, xtick_loc)
    plt.xlim(0,T)
    # plt.ylim(-1.5,1.5)
    plt.title(r'$\sigma=%1.2f$'%sigmas[noise_idx])
    # plt.xlabel(r'$t$', fontsize=18)
    # plt.ylabel(r'$x$', fontsize=18)
plt.subplot(338)
plt.xlabel(r'$t$', fontsize=18)
plt.subplot(334)
plt.ylabel(r'$x$', fontsize=18)
plt.tight_layout()
plt.savefig('dynamics')

print 'done'

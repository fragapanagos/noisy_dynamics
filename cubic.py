import numpy as np
from matplotlib import pyplot as plt
import pdb

def fun(x):
    return 8./3 * (-x**3 + x) 

T = 4
dt = 0.001 
sigmas = np.linspace(0,2,9) # noise levels

x0s = np.linspace(-1.5,1.5,10)

print 'plotting phase portrait'
x = np.linspace(-1.5,1.5,50)
plt.figure('phase')
for x0 in x0s:
    plt.plot(x0, fun(x0), 'o', markersize=10)
plt.plot(x, fun(x), color='k')
plt.axhline(color='k')
plt.xlabel(r'$x$', fontsize=18)
plt.ylabel(r'$\dot{x}$', fontsize=18)
plt.savefig('phase')

# print 'analyzing phase portrait with noise'
# plt.figure('phase_uniform_noise')
# for sigma_idx, sigma in enumerate(sigmas):
#     phase_conv = np.zeros(n)
#     for i in range(len(x)):
#         phase_conv[i] = np.mean(fun(np.linspace(x[i]- sigma / 2., x[i] + sigma / 2., 50)))
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


# print 'running dynamics...'
# plt.figure('dynamics')
# xtick_loc = [float(i) for i in range(5)]
# xtick_lab = [str(float(i)) for i in range(5)]
# for sigma_idx, sigma in enumerate(sigmas):
#     plt.subplot(3,3,sigma_idx+1)
#     for x0 in x0s:
#         now = 0
#         states = [x0]
#         time = [now]
#         state = x0
#         while now < T:
#             now = now + dt
#             state = state + fun(state + sigma * np.random.rand() - sigma / 2) * dt # uniform noise
#             time.append(now)
#             states.append(state)
#         plt.plot(time, states)
#         plt.xticks(xtick_loc, xtick_loc)
#     plt.title(r'$\sigma=%1.2f$'%sigma)
# plt.subplot(338)
# plt.xlabel(r'$x$', fontsize=18)
# plt.subplot(334)
# plt.ylabel(r'$\dot{x}$', fontsize=18)
# plt.tight_layout()
# plt.savefig('dynamics')

print 'done'

"""Contains utility functions for analyzing noisy dynamical systems"""

import numpy as np

def local_avg(x, dxdt, dist, dxmin=1, dxmax=1, npts=50):
    """Convolves dxdt with dist around x.

    Uses the region [x-dxmin: x+dxmax]"""
    if dist.sigma == 0:
        return dxdt(x, 0)
    pts, step = np.linspace(x-dxmin, x+dxmax, num=npts, retstep=True)
    return np.sum(dxdt(pts, 0) * dist.pdf(pts-x)) * step

def find_stable_pts(x, fx):
    """finds the stable points of a dynamical system"""
    pre_idx = np.where(np.diff(np.sign(fx))<-1)[0] 
    post_idx = pre_idx+1
    slopes = (fx[post_idx] - fx[pre_idx]) / (x[post_idx] - x[pre_idx])
    # interpolate to find zero crossings
    stable_x = -fx[pre_idx] / slopes + x[pre_idx] 
    return stable_x

if __name__ == "__main__":
    import pylab as plt
    x = np.arange(-1,1.1,.1)
    y = np.random.randn(len(x))
    stpts = find_stable_pts(x,y)
    plt.plot(x,y, '-o')
    plt.axhline(color='k')
    for stpt in stpts:
        plt.axvline(stpt)
    plt.show()

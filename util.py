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
    passed_zero = np.where(np.diff(np.sign(fx))<0)[0]
    stable_x = (x[passed_zero+1] + x[passed_zero]) / 2.
    return stable_x

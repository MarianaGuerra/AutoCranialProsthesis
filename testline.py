# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
    else:
        kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)


    # Calculate result
    arange = np.arange(len(u))
    points = np.zeros((len(u),cv.shape[1]))
    for i in xrange(cv.shape[1]):
        points[arange,i] = si.splev(u, (kv,cv[:,i],degree))

    return points


def main():
    gap_points = np.loadtxt("ext_control_pts_50.txt", delimiter=' ')
    gap_points = np.array(gap_points)
    points_inv = np.loadtxt("ext_inv_control_pts_50.txt", delimiter=' ')
    points_inv = np.array(points_inv)

    points = np.concatenate((gap_points, points_inv), axis=0)

    spline = bspline(points, n=300, degree=3, periodic=False)

    fig, ax = plt.subplots()
    ax.plot(gap_points[:,1], gap_points[:,0], 'm.')
    ax.plot(points_inv[:, 1], points_inv[:, 0], 'g.')
    ax.plot(spline[:,1], spline[:,0],'b-')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    plt.show()


if __name__ == '__main__':
    main()



# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
from scipy import interpolate
from scipy.interpolate import interp1d

# gap_points = np.loadtxt("ext_control_pts_50.txt", delimiter=' ')
gap_points = np.loadtxt("ext_control_pts_50_40graus.txt", delimiter=' ')
gap_points = np.array(gap_points)
points_inv = np.loadtxt("ext_inv_control_pts_50_40graus.txt", delimiter=' ')
points_inv = np.array(points_inv)
points_inv_rot = points_inv.copy()
mean_point = [227.96358683, 254.8181934]

# rotacionar inv_points
# coord centrada = coord pto - coord mp
#coord centrada * matriz rotação
#coord final = coord centrada + coord
theta = 5
theta_rad = np.deg2rad(theta)
rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])

for p in range(points_inv.shape[0]):
    centered_coord = points_inv[p] - mean_point
    rotated_coord = np.matmul(centered_coord, rot_matrix)
    final_coord = mean_point + rotated_coord
    points_inv_rot[p] = final_coord

# fig, ax = plt.subplots()
# ax.plot(points_inv_rot[:, 1], points_inv_rot[:, 0], 'm.')
# ax.plot(points_inv[:, 1], points_inv[:, 0], 'b-')

x = gap_points[:, 1]
y = gap_points[:, 0]
# s1 = np.arange(135)
# s2 = np.arange(355, 487)
# s = np.concatenate((s1,s2), axis=0)
ss = []
xs = []
ys = []

for i in range(0, 131, 10):
    ss += [i]
    xs.append(x[i])
    ys.append(y[i])

for j in range(365, 487, 10):
    ss += [j]
    xs.append(x[j-223])
    ys.append(y[j-223])

# Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’
# https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html

fx = interp1d(ss, xs, kind='quadratic')
snew = np.linspace(0, 484, num=485, endpoint=True)
plt.plot(ss, xs, 'o', snew, fx(snew), '-')
plt.legend(['data','interpolation'], loc='best')

fy = interp1d(ss, ys, kind='quadratic')
plt.plot(ss, ys, 'o', snew, fy(snew), '-')

array = np.zeros([snew.shape[0],2])
array[:,0] = fx(snew)
array[:,1] = fy(snew)

fig, ax = plt.subplots()
#ax.plot(gap_points[:, 1], gap_points[:, 0], 'm.')
ax.plot(array[:, 0], array[:, 1], 'b.')
ax.plot(fx(snew), fy(snew), 'g-')
#ax.plot(points_inv_rot[:, 1], points_inv_rot[:, 0], 'b-')
ax.set_xlim([0, 512])
ax.set_ylim([0, 512])
plt.show()





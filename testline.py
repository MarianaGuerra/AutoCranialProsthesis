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

ext_1 = np.loadtxt("ext_1_img69.txt", delimiter=' ')
ext_1 = np.array(ext_1)
sext_1 = []
for i in range(0,ext_1.shape[0], 5):
    sext_1.append(ext_1[i])
ext_1 = np.array(sext_1)

ext_2 = np.loadtxt("ext_2_img69.txt", delimiter=' ')
ext_2 = np.array(ext_2)
sext_2 = []
for i in range(0,ext_2.shape[0], 5):
    sext_2.append(ext_2[i])
ext_2 = np.array(sext_2)

points_inv_rot = points_inv.copy()
gap_points_rot = gap_points.copy()
mean_point = [227.96358683, 254.8181934]

# rotacionar inv_points

# theta = 5
# theta_rad = np.deg2rad(theta)
# rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])

# for p in range(points_inv.shape[0]):
#     centered_coord = points_inv[p] - mean_point
#     rotated_coord = np.matmul(centered_coord, rot_matrix)
#     final_coord = mean_point + rotated_coord
#     points_inv_rot[p] = final_coord
#
# for p in range(gap_points.shape[0]):
#     centered_coord = gap_points[p] - mean_point
#     rotated_coord = np.matmul(centered_coord, rot_matrix)
#     final_coord = mean_point + rotated_coord
#     gap_points_rot[p] = final_coord

# fig, ax = plt.subplots()
# ax.plot(points_inv_rot[:, 1], points_inv_rot[:, 0], 'm-')
# ax.plot(points_inv[:, 1], points_inv[:, 0], 'b-')
# ax.set_xlim([0, 512])
# ax.set_ylim([0, 512])

fig, ax = plt.subplots()
ax.plot(ext_1[:, 1], ext_1[:, 0], 'b.')
ax.plot(ext_1[0,1], ext_1[0,0], 'x')
ax.plot(ext_2[:, 1], ext_2[:, 0], 'g.')
ax.plot(ext_2[0,1], ext_2[0,0], 'x')
ax.set_xlim([0, 512])
ax.set_ylim([0, 512])
plt.show()

# testando eliminar parte de edge nos trechos ext

ang_list = []
for p in range(ext_1.shape[0] - 1):
    seg_x = ext_1[p + 1][1] - ext_1[p][1]
    seg_y = ext_1[p + 1][0] - ext_1[p][0]
    cos_ang = (seg_x*1 + seg_y*0) / (np.sqrt(seg_x**2 + seg_y**2))
    ang_list += [np.rad2deg(np.arccos(cos_ang))]
#print(str(ang_coef_list))
ang_var = np.zeros(len(ang_list) - 1)
for q in range(len(ang_list) - 1):
    ang_var[q] = (ang_list[q] - ang_list[q - 1])
# parametrizar lim usando a linha de base, mediana, algo assim
ang_var = np.convolve(ang_var, np.ones((5,))/5, mode='valid')
lim = 1.5
# find edge first segment
edge_first_seg = 0
for r in range(len(ang_var) - 2):
    if ang_var[r + 1] >= ang_var[r] + lim:
        edge_first_seg = r
        break
cutted_ext_1 = ext_1[0: edge_first_seg + 2].copy()  # not inclusive
plt.plot(range(len(ang_var)), ang_var[:], '.', edge_first_seg, ang_var[edge_first_seg], 'x')
plt.legend(['ext_1', 'edge_first_seg'], loc='best')
plt.show()


ang_list = []
for p in range(ext_2.shape[0] - 1):
    seg_x = ext_2[p + 1][1] - ext_2[p][1]
    seg_y = ext_2[p + 1][0] - ext_2[p][0]
    cos_ang = (seg_x*1 + seg_y*0) / (np.sqrt(seg_x**2 + seg_y**2))
    ang_list += [np.rad2deg(np.arccos(cos_ang))]
# print(str(ang_coef_list))
ang_var = np.zeros(len(ang_list) - 1)
for q in range(len(ang_list) - 1):
    ang_var[q] = (ang_list[q] - ang_list[q - 1])
# parametrizar lim usando a linha de base, mediana, algo assim
ang_var = np.convolve(ang_var, np.ones((5,))/5, mode='valid')
lim = 1.5
edge_final_seg = 0
for s in range(len(ang_var)-1, 0, -1):
        if ang_var[s-1] >= ang_var[s] + lim:
            edge_final_seg = s
            break
cutted_ext_2 = ext_2[edge_final_seg + 2: len(ext_2)-1].copy()  # not inclusive
plt.plot(range(len(ang_var)), ang_var[:], '.', edge_final_seg , ang_var[edge_final_seg ], 'x')
plt.legend(['ext_2', 'edge_final_seg'], loc='best')
plt.show()

fig, ax = plt.subplots()
ax.plot(cutted_ext_1[:, 1], cutted_ext_1[:, 0], 'b-')
#ax.plot(ext_1[0,1], ext_1[0,0], 'x')
ax.plot(cutted_ext_2[:, 1], cutted_ext_2[:, 0], 'g-')
#ax.plot(ext_2[0,1], ext_2[0,0], 'x')
ax.set_xlim([0, 512])
ax.set_ylim([0, 512])
plt.show()

# x = gap_points_rot[:, 1]
# y = gap_points_rot[:, 0]
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
# plt.plot(ss, xs, 'o', snew, fx(snew), '-')
# plt.legend(['data','interpolation'], loc='best')

fy = interp1d(ss, ys, kind='quadratic')
# plt.plot(ss, ys, 'o', snew, fy(snew), '-')

array = np.zeros([snew.shape[0],2])
array[:,0] = fx(snew)
array[:,1] = fy(snew)
#
# fig, ax = plt.subplots()
# # ax.plot(gap_points[:, 1], gap_points[:, 0], 'm.')
# ax.plot(gap_points_rot[:, 1], gap_points_rot[:, 0], 'm.')
# ax.plot(array[:, 0], array[:, 1], 'b-')
# # ax.plot(fx(snew), fy(snew), 'g-')
# ax.plot(points_inv_rot[:, 1], points_inv_rot[:, 0], 'g-')
# # ax.plot(points_inv[:, 1], points_inv[:, 0], 'y-')
# ax.set_xlim([0, 512])
# ax.set_ylim([0, 512])
# plt.show()





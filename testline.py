# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as si
from scipy import interpolate
from scipy.interpolate import interp1d


def sample(seg, n):
    seg = np.array(seg)
    sseg = []
    for i in range(0, seg.shape[0], n):
        sseg.append(seg[i])
    seg = np.array(sseg)
    return seg

# def rotate():
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


def find_inflection(seg, n, lim):
    ang_list = []
    for p in range(seg.shape[0]-1):
        ang = math.atan2((seg[p + 1][0] - seg[p][0]), (seg[p + 1][1] - seg[p][1]))
        ang_list += [np.rad2deg(ang)]

    ang_list = np.convolve(ang_list, np.ones((n,)) / n, mode='valid')

    plt.figure()
    plt.plot(range(len(ang_list)), ang_list[:], '.')
    plt.title("ang_list")
    plt.show()

    ang_var = np.zeros(len(ang_list) - 1)
    for q in range(len(ang_list) - 1):
        ang_var[q] = abs(ang_list[q+1] - ang_list[q])

    ang_var = np.convolve(ang_var, np.ones((n,)) / n, mode='valid')

    plt.figure()
    plt.plot(range(len(ang_var)), ang_var[:], '.')
    plt.title("ang_var")
    plt.show()

    # find edge first segment
    # edge_first_seg = 0
    # for r in range(len(ang_var) - 1):
    #     if ang_var[r + 1] >= ang_var[r] + lim:
    #         edge_first_seg = r
    #         break
    # return edge_first_seg + 1

    # mudei pra fazer a convolução dos dois pra eliminar bastante ruido e outliers
    # ai só retorno a posição do max ang_var
    # os graficos ficam bem bonitinhos, parece que funciona
    # tenta incorporar isso no resto da seleção do ext/int/toco, e vamos ver se quebra pra algum slice

    return np.argmax(ang_var)


def main():
    # gap_points = np.loadtxt("ext_control_pts_50.txt", delimiter=' ')
    gap_points = np.loadtxt("ext_control_pts_50_40graus.txt", delimiter=' ')
    gap_points = np.array(gap_points)

    points_inv = np.loadtxt("ext_inv_control_pts_50_40graus.txt", delimiter=' ')
    points_inv = np.array(points_inv)

    points_inv_rot = points_inv.copy()
    gap_points_rot = gap_points.copy()
    mean_point = [227.96358683, 254.8181934]

    # fig, ax = plt.subplots()
    # ax.plot(points_inv_rot[:, 1], points_inv_rot[:, 0], 'm-')
    # ax.plot(points_inv[:, 1], points_inv[:, 0], 'b-')
    # ax.set_xlim([0, 512])
    # ax.set_ylim([0, 512])

    ext1 = np.loadtxt("ext_1_img69.txt", delimiter=' ')
    ext1 = np.array(ext1)

    ext2 = np.loadtxt("ext_2_img69.txt", delimiter=' ')
    ext2 = np.array(ext2)

    fig, ax = plt.subplots()
    ax.plot(ext1[:, 1], ext1[:, 0], 'b.')
    ax.plot(ext1[0, 1], ext1[0, 0], 'x')
    ax.plot(ext2[:, 1], ext2[:, 0], 'g.')
    ax.plot(ext2[0, 1], ext2[0, 0], 'x')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    plt.show()

    # testando eliminar parte de edge nos trechos ext
    inf_index1 = find_inflection(ext1,15,3)
    print(str(inf_index1))
    cut_ext1 = ext1[0: inf_index1 + 2].copy()

    inf_index2 = find_inflection(ext2[::-1], 15, 3)
    print(str(inf_index2))
    cut_ext2 = ext2[inf_index2: len(ext2)+1].copy()

    fig, ax = plt.subplots()
    ax.plot(cut_ext1[:, 1], cut_ext1[:, 0], 'b-')
    ax.plot(cut_ext2[:, 1], cut_ext2[:, 0], 'g-')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    plt.show()

    # # x = gap_points_rot[:, 1]
    # # y = gap_points_rot[:, 0]
    # x = gap_points[:, 1]
    # y = gap_points[:, 0]
    # # s1 = np.arange(135)
    # # s2 = np.arange(355, 487)
    # # s = np.concatenate((s1,s2), axis=0)
    # ss = []
    # xs = []
    # ys = []
    #
    # for i in range(0, 131, 10):
    #     ss += [i]
    #     xs.append(x[i])
    #     ys.append(y[i])
    #
    # for j in range(365, 487, 10):
    #     ss += [j]
    #     xs.append(x[j-223])
    #     ys.append(y[j-223])
    #
    # # Specifies the kind of interpolation as a string (‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’
    # # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    #
    # fx = interp1d(ss, xs, kind='quadratic')
    # snew = np.linspace(0, 484, num=485, endpoint=True)
    # # plt.plot(ss, xs, 'o', snew, fx(snew), '-')
    # # plt.legend(['data','interpolation'], loc='best')
    #
    # fy = interp1d(ss, ys, kind='quadratic')
    # # plt.plot(ss, ys, 'o', snew, fy(snew), '-')
    #
    # array = np.zeros([snew.shape[0],2])
    # array[:,0] = fx(snew)
    # array[:,1] = fy(snew)
    # #
    # # fig, ax = plt.subplots()
    # # # ax.plot(gap_points[:, 1], gap_points[:, 0], 'm.')
    # # ax.plot(gap_points_rot[:, 1], gap_points_rot[:, 0], 'm.')
    # # ax.plot(array[:, 0], array[:, 1], 'b-')
    # # # ax.plot(fx(snew), fy(snew), 'g-')
    # # ax.plot(points_inv_rot[:, 1], points_inv_rot[:, 0], 'g-')
    # # # ax.plot(points_inv[:, 1], points_inv[:, 0], 'y-')
    # # ax.set_xlim([0, 512])
    # # ax.set_ylim([0, 512])
    # # plt.show()


if __name__ == '__main__':
    main()



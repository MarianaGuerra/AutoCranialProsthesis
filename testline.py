# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as si
from scipy import interpolate
from scipy.interpolate import interp1d


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


def ccw(a, b, c):
    """
    Auxiliar function to Intersect
    """
    return (c[0] - a[0]) * (b[1] - a[1]) > (b[0] - a[0]) * (c[1] - a[1])


def intersect(a, b, c, d):
    """
    Checks if two line segments AB and CD intersect 
    :param a: array, point a on line segment AB
    :param b: array, point b on line segment AB
    :param c: array, point c on line segment CD
    :param d: array, point d on line segment CD
    :return: true if line segments AB and CD intersect
    """
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def interpolation(ext_1, ext_2, spline_points_number, border_1, border_2, test_contour):
    ss = []
    xs = []
    ys = []
    # calcular as posições dos pontos de ext1 e ext2 dentro desse n
    # amostragem apenas a partir de 10 pontos de distância da extremidade da falha
    # calcular fx, fy, snew (apenas entre os 3 pontos amostra extremos da falha)
    for i in range(0, ext_1.shape[0] - border_1, 10):
        ss += [i]
        xs.append(ext_1[i, 1])
        ys.append(ext_1[i, 0])
    for j in range(spline_points_number - ext_2.shape[0] + border_2, spline_points_number, 10):
        ss += [j]
        xs.append(ext_2[j - (spline_points_number - ext_2.shape[0]), 1])
        ys.append(ext_2[j - (spline_points_number - ext_2.shape[0]), 0])
    fx = interp1d(ss, xs, kind='quadratic')
    fy = interp1d(ss, ys, kind='quadratic')
    snew = np.linspace(ext_1.shape[0] - 20, spline_points_number - ext_2.shape[0] + 20 - 1, num=j, endpoint=True)
    # calcular fx(snew), fy(snew)
    # plt.plot(ss, xs, 'o', snew, fx(snew), '-')
    # plt.plot(ss, ys, 'o', snew, fy(snew), '-')
    # plt.legend(['data', 'interpolation'], loc='best')
    # plt.show()
    # fazer array de pontos
    array = np.zeros([snew.shape[0], 2])
    array[:, 1] = fx(snew)
    array[:, 0] = fy(snew)

    fig, ax = plt.subplots()
    ax.plot(test_contour[:, 1], test_contour[:, 0], 'm-')
    ax.plot(ext_1[:, 1], ext_1[:, 0], 'b-')
    ax.plot(ext_1[0,1],ext_1[0, 0], 'x')
    ax.plot(ext_2[0, 1], ext_2[0, 0], 'x')
    ax.plot(ext_2[:, 1], ext_2[:, 0], 'b-')
    ax.plot(array[:, 1], array[:, 0], 'g-')
    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    plt.show()
    return array


def main():

    ext1 = np.loadtxt("ext_1_img69.txt", delimiter=' ')
    ext_1 = np.array(ext1)

    ext2 = np.loadtxt("ext_2_img69.txt", delimiter=' ')
    ext_2 = np.array(ext2)

    inv_ext = np.loadtxt("inv_ext_img69.txt", delimiter=' ')
    inv_ext = np.array(inv_ext)

    test_contour = np.loadtxt("contour_img69.txt", delimiter=' ')
    test_contour = np.array(test_contour)

    fig, ax = plt.subplots()
    ax.plot(test_contour[:, 1], test_contour[:, 0], 'm-')
    ax.plot(ext_1[:, 1], ext_1[:, 0], 'b-')
    ax.plot(ext_1[0, 1], ext_1[0, 0], 'x')
    ax.plot(ext_2[0, 1], ext_2[0, 0], 'x')
    ax.plot(ext_2[:, 1], ext_2[:, 0], 'b-')

    ax.set_xlim([0, 512])
    ax.set_ylim([0, 512])
    plt.show()

    # Performs interpolation to find the gap points
    spline_points_number = inv_ext.shape[0]
    border_1 = 0
    border_2 = 0
    a = 5
    b = 5
    segs = [ext_1[0]]

    ang_list = []
    for p in range(0, ext_1.shape[0] - a , a):
        seg_x = ext_1[p + a][1] - ext_1[p][1]
        seg_y = ext_1[p + a][0] - ext_1[p][0]
        cos_ang = (seg_x * 1 + seg_y * 0) / (np.sqrt(seg_x ** 2 + seg_y ** 2))
        ang_list += [np.rad2deg(np.arccos(cos_ang))]
        segs += [ext_1[p+a]]

    print(str(ext_1.shape[0]))
    print(str(ang_list))
    segs = np.array(segs)

    fig, ax = plt.subplots()
    ax.plot(ext_1[:, 1], ext_1[:, 0], 'b-')
    ax.plot(segs[:, 1], segs[:, 0], 'r.')
    plt.show()

    ang_var = []
    for q in range(b, len(ang_list)):
        ang_var += [ang_list[q] - ang_list[q-b]]

    # parametrizar lim usando a linha de base, mediana, algo assim
    #ang_var = np.convolve(ang_var, np.ones((5,)) / 5, mode='valid')
    lim = 45
    # find edge first segment
    edge_first_seg = 0
    for r in range(len(ang_var)):
        if ang_var[r] >= lim:
            edge_first_seg = r
            break
    cutted_ext_1 = ext_1[0: edge_first_seg + 2].copy()  # not inclusive
    plt.plot(range(len(ang_var)), ang_var[:], '.', edge_first_seg, ang_var[edge_first_seg], 'x')
    plt.legend(['ext_1', 'edge_first_seg'], loc='best')
    plt.show()

    # ang_list = []
    # for p in range(ext_2.shape[0] - 1):
    #     seg_x = ext_2[p + 1][1] - ext_2[p][1]
    #     seg_y = ext_2[p + 1][0] - ext_2[p][0]
    #     cos_ang = (seg_x * 1 + seg_y * 0) / (np.sqrt(seg_x ** 2 + seg_y ** 2))
    #     ang_list += [np.rad2deg(np.arccos(cos_ang))]
    # # print(str(ang_coef_list))
    # ang_var = np.zeros(len(ang_list) - 1)
    # for q in range(len(ang_list) - 1):
    #     ang_var[q] = (ang_list[q] - ang_list[q - 1])
    # # parametrizar lim usando a linha de base, mediana, algo assim
    # ang_var = np.convolve(ang_var, np.ones((5,)) / 5, mode='valid')
    # lim = 1.5
    # edge_final_seg = 0
    # for s in range(len(ang_var) - 1, 0, -1):
    #     if ang_var[s - 1] >= ang_var[s] + lim:
    #         edge_final_seg = s
    #         break
    # cutted_ext_2 = ext_2[edge_final_seg + 2: len(ext_2) - 1].copy()  # not inclusive
    # plt.plot(range(len(ang_var)), ang_var[:], '.', edge_final_seg, ang_var[edge_final_seg], 'x')
    # plt.legend(['ext_2', 'edge_final_seg'], loc='best')
    # plt.show()

    #spline_points = interpolation(ext1, ext2, spline_points_number, border_1, border_2, test_contour)


if __name__ == '__main__':
    main()



# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.mlab import PCA
# from open import load_dicom_folder, dicom_datasets_to_numpy
# from skimage import measure
# from scipy.spatial import distance
#
# import scipy.optimize as optimize
# import mpl_toolkits.mplot3d as m3d
# import scipy.optimize
# from mpl_toolkits.mplot3d import Axes3D


def registration(contour1, contour2):
    """
    Calculates registration
    :param contour1: 
    :param contour2: 
    :return: 
    """
    # calculate the centroids of both contours
    centroid1 = (1/np.shape(contour1)[0])*np.sum(contour1, axis=0)
    print("centroid 1 = " + str(centroid1))
    centroid2 = (1 / np.shape(contour2)[0]) * np.sum(contour2, axis=0)
    print("centroid 2 = " + str(centroid2))

    # not necessary to put both centroids on origin because they have the same reference
    H = np.array([[0, 0]])
    if contour1.shape[0] < contour2[0].shape[0]:
        size = contour2[0].shape[0]
    else:
        size = contour1[0].shape[0]
    for x in range(size):
        # atenção para tamanho da matriz a, deve ser 2x2 ou 3x3 => pontos devem ser transpostos para este resultado
        a = np.matmul(np.transpose(np.matrix(contour1[x] - centroid1)), np.matrix(contour2[x] - centroid2))
        print("a = " + str(a))
        H = np.add(H, a)
        print("H = " + str(H))
    print("H final = " + str(H))
    U, s, V = np.linalg.svd(H)
    R = V * np.transpose(U)
    return R


def main():
    # contorno1 = np.array([[1, 1], [1.5, 2.5], [3, 2.75], [4, 2.5], [5, 1.5]])
    # contorno2 = np.array([[1.5, 1.5], [2, 3], [3.5, 3.25], [4.5, 3], [5.5, 2]])
    # rot_matrix = np.array([[np.cos(np.pi/9), -np.sin(np.pi/9)],
    #                        [np.sin(np.pi/9), np.cos(np.pi/9)]])
    # rot_matrix2 = np.array([[np.cos(-np.pi / 9), -np.sin(-np.pi / 9)],
    #                        [np.sin(-np.pi / 9), np.cos(-np.pi / 9)]])
    # # rot_matrix2: transforma contorno 3 de volta no 1
    # # array([[ 0.93969262,  0.34202014],
    # #       [-0.34202014,  0.93969262]])
    # contorno3 = np.matmul(contorno1, rot_matrix)
    # contorno4 = np.matmul(contorno3, rot_matrix2)
    # fig = plt.figure()
    # plt.plot(contorno1[:, 0], contorno1[:, 1], c='red')
    # plt.plot(contorno2[:, 0], contorno2[:, 1], c='blue')
    # # plt.plot(contorno3[:, 0], contorno3[:, 1], c='blue')

    side_a = np.loadtxt("a_cont40.txt", delimiter=' ')
    side_b = np.loadtxt("b_mirror_cont40.txt", delimiter=' ')
    fig = plt.figure()
    plt.scatter(side_a[:, 0], side_a[:, 1], c='red')
    plt.scatter(side_a[0, 0], side_a[0, 1], c='black')
    # plt.plot(side_a[:, 0], side_a[:, 1], c='red')
    plt.scatter(side_b[:, 0], side_b[:, 1], c='blue')
    plt.scatter(side_b[0, 0], side_b[0, 1], c='green')

    # R = registration(side_a, side_b)
    # print("rot matrix = " + str(R))
    #
    # result = np.matmul(side_b, R)
    # fig2 = plt.figure()
    # plt.scatter(result[:, 0], result[:, 1], c='black')
    # plt.plot(contorno4[:, 0], contorno4[:, 1], c='black')

    plt.show()


if __name__ == '__main__':
    main()


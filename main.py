# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
from open import load_dicom_folder, dicom_datasets_to_numpy
from skimage import measure
from scipy.spatial import distance
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
import scipy.interpolate as si
import random
from mpl_toolkits.mplot3d import Axes3D


# Mariana Guerra
# Cranial prosthesis modeling


def select_contours(img):
    """
    Evaluates all contour found to select only the ones centered near the image center
    :param img: 2D ndarray of DICOM image converted by dicom_datasets_to_numpy
    :return: list with the wanted contours; list with the central pixel of each wanted contour
    """
    # Find contours at a constant value
    contours = measure.find_contours(img, 200)
    # print("Found " + str(len(contours)) + " contour(s)")
    # Select the nearest contours with respect to the center pixel of the image
    width = img.shape[1]  # number of columms
    heigth = img.shape[0]  # number of rows
    pixel_ref = (width / 2, heigth / 2)
    # Threshold distance is 10% of images smallest dimension
    dist_thresh = min(width, heigth) * 0.1
    contours_wanted = []
    pixel_mean_array = []
    for contour in contours:
        if len(contour) > 250:
            contour_3d = np.zeros([contour.shape[0], 3])  # 3rd dimension added for later conversion to pat coord space
            contour_3d[:, :2] = contour
            pixel_mean = np.mean(contour, axis=0)
            if distance.euclidean(pixel_ref, pixel_mean) <= dist_thresh:
                contours_wanted.append(contour_3d)
                pixel_mean_array.append(pixel_mean)
    print("Set " + str(len(contours_wanted)) + " contour(s)")
    return contours_wanted, pixel_mean_array


def calculate_line_from_points(mpr):
    """
    Calculates point and direction to define a line on 3D coordinate space based on array of points via single value
    decompostion (SVD)
    :param mpr: ndarray of points to them the line will fit
    :return: 3D ndarray vector that defines the line; 3D ndarray point on line 
    """
    mean = np.mean(mpr, 0)
    mpr_sub = copy.deepcopy(mpr)
    mpr_sub[:] = [x - mean for x in mpr_sub]
    u, s, v = np.linalg.svd(mpr_sub)
    direction = v[0, :]
    direction = direction / np.sqrt(direction.dot(direction))
    return direction, mean


def point_on_line(point, direction, z):
    """
    Finds x and y coordinate of a wanted point on line given point and direction and wanted point z coordinate
    :param point: 3D ndarray point on line 
    :param direction: 3D ndarray vector that defines the line
    :param z: z coordinate of the wanted point ( = slice)
    :return: 3D ndarray with x, y and z coordinates of the wanted point
    """
    # s*direction + point = [x, y, z]
    s = (z - point[2])/direction[2]
    x = point[0] + s * direction[0]
    y = point[1] + s * direction[1]
    return np.array([x, y, z])


def invert_point(point, ref_point):
    """
    Inverts a point with respect to a line formed by a reference vector and a reference point
    :param point: original/input point
    :param ref_point: reference point for the line
    :return: inverted point
    """
    ref_point = ref_point[0:2]
    ref_vector = np.array([1, 0])
    inverted_point = ref_point - (point - ref_point) + 2 * ref_vector * np.dot((point - ref_point), ref_vector)
    return inverted_point


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


def intersect_contour(test_contour, mid_point, theta):
    ref_vector = np.array([1, 0])
    theta = np.deg2rad(theta)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    ref_vec = np.matmul(ref_vector, rot_matrix)
    far_point = mid_point + 300 * ref_vec
    intersected = []
    intersected_index = []
    for p in range(test_contour.shape[0] - 1):
        # Return true if line segments AB and CD intersect
        if intersect(mid_point, far_point, test_contour[p], test_contour[p + 1]):
            intersected += [test_contour[p], test_contour[p + 1]]
            intersected_index += [p, p+1]
    if len(intersected) == 0:
        return None
    return intersected_index


def intersect_trough_angles(test_contour, mid_point, ang_min, ang_max, step):
    for theta in range(ang_min, ang_max, step):
        intersected = intersect_contour(test_contour, mid_point, theta)
        if intersected is None:
            return theta


def find_gap_angles(mid_point, test_contour):
    # Searching clockwise for the first angle with no intersection = first gap edge
    theta_1 = intersect_trough_angles(test_contour, mid_point, 0, 360, 30)
    print ("Theta 1 to 6")
    print (str(theta_1))
    # Refining
    theta_2 = intersect_trough_angles(test_contour, mid_point, theta_1 - 30, theta_1, 5)
    print (str(theta_2))
    # Refining
    theta_3 = intersect_trough_angles(test_contour, mid_point, theta_2 - 15, theta_2, 1)
    print (str(theta_3))
    # Searching counterclockwise for the first angle with no intersection = second gap edge
    theta_4 = intersect_trough_angles(test_contour, mid_point, 360, 0, -30)
    print (str(theta_4))
    # Refining
    theta_5 = intersect_trough_angles(test_contour, mid_point, theta_4 + 30, theta_4, -5)
    print (str(theta_5))
    # Refining
    theta_6 = intersect_trough_angles(test_contour, mid_point, theta_5 + 15, theta_5, -1)
    print (str(theta_6))
    # gap_angles = [theta_3, theta_6]
    gap_angles = [np.deg2rad(theta_3), np.deg2rad(theta_6)]
    return theta_3, theta_6


def plot_contours(img, contours, mean_point):
    # Display the image and plot all contours in a array of contours
    fig, ax = plt.subplots()
    contour_img = ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray, origin='bottom')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)  # x and y are switched for correct image plot
        ax.plot(mean_point[1], mean_point[0], 'ro')
    ax.axis('image')
    plt.colorbar(contour_img, ax=ax)
    plt.show()


def plot_inverted_contours(img, inverted_contours, contours, mean_point):
    # Display the image and plot all contours in a array of contours
    # x and y are switched everywhere for correct image plot
    fig, ax = plt.subplots()
    contour_img = ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray, origin='bottom')
    for inv_contour in inverted_contours:
        ax.plot(inv_contour[:, 1], inv_contour[:, 0], linewidth=2)  # x and y are switched for correct image plot
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], 'r--')
        ax.plot(mean_point[1], mean_point[0], 'ro')
        p1 = mean_point[0:2] + np.multiply(250, [1, 0])
        p2 = mean_point[0:2] - np.multiply(250, [1, 0])
        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], linewidth=2)
    ax.axis('image')
    plt.colorbar(contour_img, ax=ax)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.showMaximized()
    plt.show()


def main():
    # datasets = load_dicom_folder(r"C:\Users\Escritorio\Dropbox\USP\Projeto Mariana\TestSeries\JLL")
    datasets = load_dicom_folder(r"C:\Users\Escritorio\Dropbox\USP\Projeto Mariana\TestSeries\nic2")  # Nic
    # datasets = load_dicom_folder(r"C:\Users\Escritorio\Dropbox\USP\Projeto Mariana\TestSeries\D10A2878") #Darci
    series_arr, _ = dicom_datasets_to_numpy(datasets)
    num_images = series_arr.shape[2]
    contours_list = [None] * num_images  # list of all contours of all slices
    contours_mean_point_list = np.zeros((num_images, 3))  # list of all mean points of contours of interest
    healthy_mean_points = []  # to storage points on the skull axis line (healthy slices)
    gap_mean_points = []  # to points on the skull axis line (bone missing slices)
    healthy_slices = []
    gap_slices = []

    for i in range(num_images):
        img = series_arr[:, :, i]
        print("Image " + str(i))
        [cw, pma] = select_contours(img)  # returns contours_wanted and pixel_mean_array
        if len(cw) == 0:
            print("It wasn't possible to set a contour for this slice. \ "
                  "Please check threshold values in function 'select_contours'")
            return
        # Healthy skull slice has outside and inside contours (pixel_mean_array has 2 points)
        # Shorter contour =[0], longuer contour =[1])
        if len(pma) == 2:
            contour_0_len = len(cw[0])
            contour_1_len = len(cw[1])
            if contour_0_len >= contour_1_len:
                cw[0], cw[1] = cw[1], cw[0]
                pma[0], pma[1] = pma[1], pma[0]
            # Sets the mean point of the shorter contour as mean point(contours are approx. concentric and avoids
            # deviations caused by the points os face bone
            mean_point = list(pma[0]) + [i]
            healthy_mean_points += [mean_point]
            healthy_slices += [i]
            contours_mean_point_list[i] = mean_point
        contours_list[i] = cw

    # Calculates direction and mean point to define skull axial axis
    direction, mean = calculate_line_from_points(healthy_mean_points)

    # Calculates contour mean point for bone missing skull slices using skull axial axis
    for j in range(num_images):
        if len(contours_list[j]) == 1:  # bone missing skull slice has only one contour
            mean_point = point_on_line(mean, direction, j)
            gap_mean_points += [mean_point]
            gap_slices += [j]
            contours_mean_point_list[j] = mean_point
            # plot_contours(series_arr[:, :, j], contours_list[j], mean_point)
        else:
            pass
            # plot_contours(series_arr[:, :, j], contours_list[j], contours_mean_point_list[j])

    # Plots in blue central contour points of healthy slices (ref points for axial axis), plots in red central contour
    # points calculated for bone missing slices, plots all contours from contours_list
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # hmpa = np.asarray(healthy_mean_points)
    # gmpa = np.asarray(gap_mean_points)
    # ax.scatter(hmpa[:, 0], hmpa[:, 1], hmpa[:, 2])
    # ax.scatter(gmpa[:, 0], gmpa[:, 1], gmpa[:, 2], c='red')
    # for k in range(num_images):
    #     for contour in contours_list[k]:
    #         ax.plot(contour[:, 0], contour[:, 1], k, linewidth=1)
    # ax.set_xlim3d(0, 512)
    # ax.set_ylim3d(0, 512)
    # ax.set_zlim3d(0, num_images)
    # plt.axis('scaled')
    # plt.show()

    # Inverts contours
    inverted_contours_list = copy.deepcopy(contours_list)
    for m in range(num_images):
        for contour in inverted_contours_list[m]:
            contour_2d = contour[:, :2]
            for n in range(contour.shape[0]):
                contour_2d[n] = invert_point(contour_2d[n], contours_mean_point_list[m])
                contour[:, :2] = contour_2d
        # plot_inverted_contours(series_arr[:, :, m], \
        #                       inverted_contours_list[m], contours_list[m], contours_mean_point_list[m])

    # Determines gap region on contour
    # fazer um for pelos contornos com falha usando gap_slices, onde está 50 substituir por n
    test_contour = contours_list[50][0][:, :2]
    mid_point = contours_mean_point_list[50][0:2]
    theta_3, theta_6 = find_gap_angles(mid_point, test_contour)
    # print("Gap angles " + str(theta_3) + ", " + str(theta_6))

    # Performs contour cutting  ATENÇÃO: PODE NÃO VALER PARA FALHAS NO LADO ESQUERDO - investigar
    # Setting cut angles based on the gap angles determined above
    cut_angle_a = theta_3 - 40
    cut_angle_b = theta_6 + 40

    # Separates inverted contour in parts: internal, external
    inv_test_contour = inverted_contours_list[50][0][:, :2]
    cut_points_a = intersect_contour(inv_test_contour, mid_point, cut_angle_a)
    cut_points_b = intersect_contour(inv_test_contour, mid_point, cut_angle_b)
    inv_ext = inv_test_contour[cut_points_b[0]:cut_points_a[1]].copy()
    inv_int = inv_test_contour[cut_points_a[2]:cut_points_b[3]].copy()

    # Separates contour in parts: internal, external, gap edges
    cut_points_1 = intersect_contour(test_contour, mid_point, cut_angle_a)
    cut_points_2 = intersect_contour(test_contour, mid_point, cut_angle_b)

    contour_edge1 = test_contour[cut_points_1[0]:cut_points_1[3]].copy()
    contour_edge2 = test_contour[cut_points_2[0]:cut_points_2[3]].copy()

    # contour_edge_1: amarelo, 1os ptos são externos
    # contour_edge_2: magenta, 1os ptos são internos
    edge_points_1 = intersect_contour(contour_edge1, mid_point, theta_3 - 3)
    edge_points_2 = intersect_contour(contour_edge2, mid_point, theta_6 + 3)

    # edge 1
    ext_1 = contour_edge1[0:edge_points_1[1]].copy()
    edge_1 = contour_edge1[edge_points_1[1]:edge_points_1[2]+1].copy()
    int_1 = contour_edge1[edge_points_1[2]+1: len(contour_edge1) - 1].copy()

    # edge 2
    int_2 = contour_edge2[0:edge_points_2[1]].copy()
    edge_2 = contour_edge2[edge_points_2[1]:edge_points_2[2] + 1].copy()
    ext_2 = contour_edge2[edge_points_2[2] + 1: len(contour_edge2) - 1].copy()

    # Performs interpolation to find the gap points
    # pegar n pontos contorno invertido
    spline_points_number = inv_ext.shape[0]
    ss = []
    xs = []
    ys = []

    # calcular as posições dos pontos de ext1 e ext2 dentro desse n
    # amostragem apenas a partir de 10 pontos de distância da extremidade da falha
    # calcular fx, fy, snew (apenas entre os 3 pontos amostra extremos da falha)

    for i in range(10, ext_1.shape[0] - 10, 10):
        ss += [i]
        xs.append(ext_1[i, 1])
        ys.append(ext_1[i, 0])

    for j in range(spline_points_number - ext_2.shape[0] + 10, spline_points_number - 10, 10):
        ss += [j]
        xs.append(ext_2[j - (spline_points_number - ext_2.shape[0]), 1])
        ys.append(ext_2[j - (spline_points_number - ext_2.shape[0]), 0])

    fx = interp1d(ss, xs, kind='quadratic')
    fy = interp1d(ss, ys, kind='quadratic')

    snew = np.linspace(ext_1.shape[0] - 40, spline_points_number - ext_2.shape[0] + 40 - 1, num=j, endpoint=True)

    # calcular fx(snew), fy(snew)
    plt.plot(ss, xs, 'o', snew, fx(snew), '-')
    plt.plot(ss, ys, 'o', snew, fy(snew), '-')
    plt.legend(['data', 'interpolation'], loc='best')
    plt.show()

    # fazer array de pontos
    # colocar esse array em estrutura (lista) com número de slices com falha no slice correspondente


if __name__ == '__main__':
    main()

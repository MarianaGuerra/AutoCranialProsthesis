# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
from open import load_dicom_folder, dicom_datasets_to_numpy
from skimage import measure
from scipy.spatial import distance
import matplotlib.pyplot as plt
import copy
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
    print("Set " + str(len(contours_wanted)) + " contours of interest")
    return contours_wanted, pixel_mean_array


def calculate_line_from_points(mpr):
    """
    Calculates point and direction to define a line on 3D coordinate space based on array of points via single value
    decompostion (SVD)
    :param mpr: ndarray of points to them the line will fit
    :return: 3D ndarray vector that defines the line; 3D ndarray point on line 
    """
    mean = np.mean(mpr, 0)
    mpr_sub = copy.copy(mpr)
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
    healthy_mean_points = [0, 0, 0]  # to storage points on the skull axis line (healthy slices)
    gap_mean_points = [0, 0, 0]  # to points on the skull axis line (bone missing slices)

    for i in range(num_images):
        img = series_arr[:, :, i]
        [cw, pma] = select_contours(img)  # returns contours_wanted and pixel_mean_array
        # Healthy skull slice has outside and inside contours (pixel_mean_array has 2 points)
        # Setting which one is the internal / external contour (internal=[0], external=[1]) when needed
        if len(pma) == 2:
            contour_0_len = len(cw[0])
            contour_1_len = len(cw[1])
            if contour_0_len >= contour_1_len:
                cw[0], cw[1] = cw[1], cw[0]
            # Sets the mean point of the external contour (contours are approx. concentric) as mean point
            mean_point = list(pma[1]) + [i]
            healthy_mean_points = np.vstack([healthy_mean_points, mean_point])
            contours_mean_point_list[i] = mean_point
        contours_list[i] = cw

    # Calculates direction and mean point to define skull axial axis
    healthy_mean_points = healthy_mean_points[1:, :]  # first point was 0 for inicialization only
    direction, mean = calculate_line_from_points(healthy_mean_points)

    # Calculates contour mean point for bone missing skull slices using skull axial axis
    for j in range(num_images):
        if len(contours_list[j]) == 1:  # bone missing skull slice has only one contour
            mean_point = point_on_line(mean, direction, j)
            gap_mean_points = np.vstack([gap_mean_points, mean_point])
            contours_mean_point_list[j] = mean_point
            # plot_contours(series_arr[:, :, j], contours_list[j], mean_point)
        else:
            pass
            # plot_contours(series_arr[:, :, j], contours_list[j], contours_mean_point_list[j])
    gap_mean_points = gap_mean_points[1:, :]  # first point was 0 for inicialization only

    # Plots in blue central contour points of healthy slices (ref points for axial axis), plots in red central contour
    # points calculated for bone missing slices, plots all contours from contours_list
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(healthy_mean_points[:, 0], healthy_mean_points[:, 1], healthy_mean_points[:, 2])
    # ax.scatter(gap_mean_points[:, 0], gap_mean_points[:, 1], gap_mean_points[:, 2], c='red')
    # for k in range(num_images):
    #     for contour in contours_list[k]:
    #         ax.plot(contour[:, 0], contour[:, 1], k, linewidth=1)
    # ax.set_xlim3d(0, 512)
    # ax.set_ylim3d(0, 512)
    # ax.set_zlim3d(0, num_images)
    # plt.axis('scaled')
    # # p1 = point_on_line(mean, direction, 160)  # reference point for line plot
    # # p2 = point_on_line(mean, direction, 240)  # reference point for line plot
    # # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])  # format: [x1, x2] [y1, y2] [z1, z2]
    # plt.show()

    # Invert contours
    inverted_contours_list = copy.deepcopy(contours_list)
    for m in range(num_images):
        for contour in inverted_contours_list[m]:
            contour_2d = contour[:, :2]
            for n in range(contour.shape[0]):
                contour_2d[n] = invert_point(contour_2d[n], contours_mean_point_list[m])
                contour[:, :2] = contour_2d
        plot_inverted_contours(series_arr[:, :, m], inverted_contours_list[m], contours_list[m], contours_mean_point_list[m])


# def contours_to_patient_coord_sys(series_arr, datasets, contours_list, contours_mean_point_list):  # reformar
#     """
#     Transforms the contours to patient coordinate system and stores them in contours_list
#     :param datasets: loaded DICOM images by load_dicom_folder
#     :param series_arr: 3D ndarray of DICOM image series converted by dicom_datasets_to_numpy
#     :return: contours_list: list of lists of 3D ndarrays (contours) for every slice, on patient coord system
#              mean_points_real: 3D ndarray of mean points of healthy skull slices on patient coord system
#              contours_mean_point_list: list of the mean point of one contour for each slice
#     """
#     contours_list_real = [None] * series_arr.shape[2]  # list of all contours of all slices
#     contours_mean_point_list_real = [None] * series_arr.shape[2]  # list of all mean points of contours of interest
#
#     # Converts all contours for patient coordinate space based on DICOM tag information
#     for i in range(series_arr.shape[2]):
#         # Collecting image information
#         img_orient_pat = [float(x) for x in list(datasets[i].ImageOrientationPatient)]
#         img_position_pat = [float(x) for x in list(datasets[i].ImagePositionPatient)]
#         pixel_spacing = [float(x) for x in list(datasets[i].PixelSpacing)]
#         iop1 = np.array(img_orient_pat[0:3])
#         iop2 = np.array(img_orient_pat[3:6])
#
#         # Coordinate system conversion for all contours
#         for contour in contours_list[i]:
#             for k in range(len(contour)):
#                 contour[k] = img_position_pat \
#                              + iop1 * pixel_spacing[1] * contour[k][0] \
#                              + iop2 * pixel_spacing[0] * contour[k][1]
#         contours_list_real[i] = cw_real
#         # Collecting points to skull axial axis and lateral inversion calculation
#         if len(pma) == 2:  # healthy skull slice has outside and inside contours (pixel_mean_array has 2 points)
#             # uses the mean point of the external contour (contours are approx. concentric)
#             pixel_mean_real = img_position_pat \
#                               + iop1 * pixel_spacing[1] * pma[1][0] \
#                               + iop2 * pixel_spacing[0] * pma[1][1]
#             contours_mean_point_list[i] = pixel_mean_real
#             mean_points_list_real = np.vstack([mean_points_real, pixel_mean_real])
#
#     return contours_list_real, mean_points_list_real


if __name__ == '__main__':
    main()

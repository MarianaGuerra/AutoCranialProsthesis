# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
from open import load_dicom_folder, dicom_datasets_to_numpy
from icp import icp_wrap
from skimage import measure
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
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
    contours = measure.find_contours(img, 300)
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
        contour_3d = np.zeros([contour.shape[0], 3])  # 3rd dimension added for later conversion to patient coord space
        contour_3d[:, :2] = contour
        pixel_mean = np.mean(contour, axis=0)
        if distance.euclidean(pixel_ref, pixel_mean) <= dist_thresh:
            contours_wanted.append(contour_3d)
            pixel_mean_array.append(pixel_mean)
    # print("Set " + str(len(contours_wanted)) + " contours of interest")
    return contours_wanted, pixel_mean_array


def contours_to_patient_coord_sys_and_points_to_skull_axial_axis(datasets, series_arr):
    """
    Transforms the contours to patient coordinate system and stores them in contours_list 
    :param datasets: loaded DICOM images by load_dicom_folder
    :param series_arr: 3D ndarray of DICOM image series converted by dicom_datasets_to_numpy
    :return: contours_list: list of lists of 3D ndarrays (contours) for every slice, on patient coord system
             mean_points_real: 3D ndarray of mean points of healthy skull slices on patient coord system
             contours_mean_point_list: list of the mean point of one contour for each slice
    """
    mean_points_real = [0, 0, 0]  # to storage points on the skull axis line (healthy slices)
    contours_list = [None] * series_arr.shape[2]  # list of all contours of all slices
    contours_mean_point_list = [None] * series_arr.shape[2]  # list of all mean points of contours of interest
    rotation_info_list = []  # to storage rotation info found by the icp
    # Converts all contours for patient coordinate space based on DICOM tag information
    for i in range(series_arr.shape[2]):
        img = series_arr[:, :, i]
        # Collecting image information
        img_orient_pat = [float(x) for x in list(datasets[i].ImageOrientationPatient)]
        img_position_pat = [float(x) for x in list(datasets[i].ImagePositionPatient)]
        pixel_spacing = [float(x) for x in list(datasets[i].PixelSpacing)]
        iop1 = np.array(img_orient_pat[0:3])
        iop2 = np.array(img_orient_pat[3:6])
        # Finding contours
        [cw, pma] = select_contours(img)
        # Setting which one is the internal / external contour (internal=[0], external=[1]) when needed
        if len(pma) == 2:
            contour_0_len = len(cw[0])
            contour_1_len = len(cw[1])
            if contour_0_len >= contour_1_len:
                cw[0], cw[1] = cw[1], cw[0]
        cw_real = copy.copy(cw)
        # Coordinate system conversion for all contours
        for contour in cw_real:
            for k in range(len(contour)):
                contour[k] = img_position_pat \
                             + iop1 * pixel_spacing[1] * contour[k][0] \
                             + iop2 * pixel_spacing[0] * contour[k][1]
        contours_list[i] = cw_real
        # Collecting points to skull axial axis and lateral symmetry calculation
        if len(pma) == 2:  # healthy skull slice has outside and inside contours (pixel_mean_array has 2 points)
            # uses the mean point of the external contour (contours are approx. concentric)
            pixel_mean_real = img_position_pat \
                              + iop1 * pixel_spacing[1] * pma[1][0] \
                              + iop2 * pixel_spacing[0] * pma[1][1]
            contours_mean_point_list[i] = pixel_mean_real
            mean_points_real = np.vstack([mean_points_real, pixel_mean_real])
            # Lateral symmetry
            # external_contour_mirrored = mirror_contour_point(cw_real[1][:, 0:2], pixel_mean_real[0:2])
            # T = icp_wrap(cw_real[1][:, 0:2], external_contour_mirrored, debug=True)
            # rotation_info_list.append(T)

    return contours_list


def plot_contours(img, contours):
    # Display the image and plot all contours in a array of contours
    fig, ax = plt.subplots()
    contour_img = ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray, origin='bottom')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)  # x and y are switched for correct image plot
    ax.axis('image')
    plt.colorbar(contour_img, ax=ax)
    plt.show()


def main():
    datasets = load_dicom_folder(r"C:\Users\Escritorio\Dropbox\USP\Projeto Mariana\TestSeries\JLL")
    series_arr, _ = dicom_datasets_to_numpy(datasets)

    contours_list = \
        contours_to_patient_coord_sys_and_points_to_skull_axial_axis(datasets, series_arr)

    # Plots all contours from contours_list
    fig, ax = plt.subplots()
    contour = contours_list[3][0]
    ax.plot(contour[:, 0], contour[:, 1], linewidth=1)
    tck, u = splprep(contour.T, u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    ax.plot(x_new, y_new, 'b--')

    # for j in range(len(contours_list)):
    #     for contour in contours_list[j]:
    #         ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], linewidth=1)
    # ax.set_xlim3d(-200, 200)
    # ax.set_ylim3d(-50, 50)
    # ax.set_zlim3d(100, 200)
    # plt.axis('scaled')

    plt.show()


if __name__ == '__main__':
    main()
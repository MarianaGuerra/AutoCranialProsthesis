# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
from open import load_dicom_folder, dicom_datasets_to_numpy
from skimage import measure
from scipy.spatial import distance
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D


# Mariana Guerra


def select_contours(img):
    """
    Evaluates all contour found to select only the ones centered near the image center
    :param img: 2D ndarray of DICOM image converted by dicom_datasets_to_numpy
    :return: list with the wanted contours; list with the central pixel of each wanted contour
    """
    # Find contours at a constant threshold value
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
    print("  Set " + str(len(contours_wanted)) + " contour(s)")
    return contours_wanted, pixel_mean_array


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


def main():
    datasets = load_dicom_folder(r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\98890234_20030505_CT.tar\98890234_20030505_CT\98890234\20030505\CT\CT3c")  # Nic
    series_arr, _ = dicom_datasets_to_numpy(datasets)
    num_images = series_arr.shape[2]
    contours_list = [None] * num_images  # list of all contours of all slices

    for i in range(num_images):
        img = series_arr[:, :, i]
        print("Image " + str(i))
        [cw, pma] = select_contours(img)  # returns contours_wanted and pixel_mean_array
        if len(cw) == 0:
            print("It wasn't possible to set a contour for this slice. \ "
                  "Please check threshold values in function 'select_contours'")
            return
        # Healthy skull slice has outside and inside contours (pixel_mean_array has 2 points)
        # Shorter contour =[0], longer contour =[1])
        if len(pma) == 2:
            contour_0_len = len(cw[0])
            contour_1_len = len(cw[1])
            if contour_0_len >= contour_1_len:
                cw[0], cw[1] = cw[1], cw[0]
                pma[0], pma[1] = pma[1], pma[0]
        contours_list[i] = cw
    print("Contours list done")

    # np.savetxt("contourimg14.txt", contours_list[14][1], delimiter=' ')
    # Creating phanton with gaps on slices 05 to 10
    gap_contours_list = copy.deepcopy(contours_list)
    round = [0, 20, 40, 60, 40, 20]
    for j in range(5, 11):
        c_falha = gap_contours_list[j][1]
        gap_size = c_falha.shape[0]/14
        gap_ini = 2*gap_size + gap_size - round[j-5]
        gap_end = gap_ini + gap_size + 2*round[j-5]
        c_falha = np.concatenate((c_falha[0:gap_ini, :], c_falha[gap_end:c_falha.shape[0], :]), axis=0)
        gap_contours_list[j][1] = c_falha

    fig = plt.figure()
    ax = Axes3D(fig)
    # hmpa = np.asarray(healthy_mean_points)
    # gmpa = np.asarray(gap_mean_points)
    # ax.scatter(hmpa[:, 0], hmpa[:, 1], hmpa[:, 2], c='green')
    # ax.scatter(gmpa[:, 0], gmpa[:, 1], gmpa[:, 2], c='red')
    for k in range(1, 14):
        contour = gap_contours_list[k][1]
        ax.plot(contour[:, 0], contour[:, 1], k, 'bo', markersize=1, alpha=0.5)
        ax.scatter(contour[0, 0], contour[0, 1], k, c='red')
    ax.set_xlim3d(0, 512)
    ax.set_ylim3d(0, 512)
    ax.set_zlim3d(0, num_images)
    plt.axis('scaled')
    plt.show()


if __name__ == '__main__':
    main()

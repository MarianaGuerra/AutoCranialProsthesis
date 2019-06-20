# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
from open import load_dicom_folder, dicom_datasets_to_numpy, PatientSpaceConversion
from skimage import measure
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.spatial.distance import directed_hausdorff


# Mariana Guerra
# Methodology for semiautomatic generation of cranial prosthesis three-dimensional model


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
    # print("  Set " + str(len(contours_wanted)) + " contour(s)")
    return contours_wanted, pixel_mean_array


def create_phantom(contours_list, center, radius, converter):
    """
    Creates phanton deleting points whithin the volume of a sphere
    :param contours_list: list with patient bone contours
    :param center: chosen point to be the sphere center
    :param radius: sphere radius
    :param converter: instance of PatientSpaceConversion
    :return: list with patient bone contours with gap determined by sphere's dimentions
    """
    gap_contours_list = [None] * len(contours_list)
    (center_ps_x, center_ps_y, center_ps_z) = converter.convert(center[0], center[1], center[2])  # ps = patient space
    for j in range(len(contours_list)):
        contour = contours_list[j][1]  # external contour
        gap_contour = []
        for i in range(contour.shape[0]):
            p = contour[i]
            (p_ps_x, p_ps_y, p_ps_z) = converter.convert(p[0], p[1], j)
            p_ps = (p_ps_x, p_ps_y, p_ps_z)
            d = (center_ps_x - p_ps_x)**2 + (center_ps_y - p_ps_y)**2 + (center_ps_z - p_ps_z)**2
            if d >= radius**2:
                gap_contour.append(p_ps)
        gap_contours_list[j] = gap_contour
    return gap_contours_list


def mirror_contours(phantom):
    """
    Mirrors contours with respect to a reference axis (reference point, horizontal vector)
    :param phantom: list with patient bone contours with gap determined by the create_phantom function
    :return: list with patient bone contours with gap mirrored, list of the reference points used for each contour
    """
    mirrored_phantom = copy.deepcopy(phantom)
    ref_point_list = [None] * mirrored_phantom.shape[0]
    for j in range(mirrored_phantom.shape[0]):
        contour = np.asarray(mirrored_phantom[j])
        # Find x and y min and max values to calculate reference point
        xmin = np.amin(contour[:, 0])
        xmax = np.amax(contour[:, 0])
        x_mean_axis = (xmin + xmax) / 2
        ymin = np.amin(contour[:, 1])
        ymax = np.amax(contour[:, 1])
        y_mean_axis = (ymin + ymax) / 2
        # Calculate reference point
        ref_point = np.asarray([x_mean_axis, y_mean_axis])
        # Create the mirrored gap contour with respect to the x axis and then translate it 2*y of ref_point
        for m in range(0, contour.shape[0]):
            point = contour[m]
            point[1] = - point[1] + 2 * ref_point[1]
        mirrored_phantom[j] = contour
        ref_point_list[j] = ref_point
    return mirrored_phantom, ref_point_list


def find_hemisphere(contours_list, phantom, mirrored_phantom, ref_point_list, converter):
    """
    Finds the hemisphere of interest (the one with the gap)
    :param contours_list: list with patient bone contours
    :param phantom: list with patient bone contours with gap determined by the create_phantom function
    :param mirrored_phantom: list with patient bone contours with gap mirrored by the mirror_contours function
    :param ref_point_list: list of the reference points used for each contour
    :param converter: instance of PatientSpaceConversion
    :return: the hemisphere of interest of the gold standard, phantom and mirrored phantom datasets
    """
    num_images = len(contours_list)
    hemi_gold_standard = [None] * num_images
    hemi_phantom = [None] * num_images
    hemi_mirrored_phantom = [None] * num_images

    # Finds out if the majority of contours in phantom have more points above or under the ref axis
    aboveaxis = 0
    underaxis = 0
    for i in range(num_images):
        ref_point = ref_point_list[i]
        c_gap = np.asarray(phantom[i])
        c_gap_aboveaxis = 0
        c_gap_underaxis = 0
        for n in range(0, c_gap.shape[0]):
            point = c_gap[n]
            if point[1] < ref_point[1]:
                c_gap_underaxis += 1
            else:
                c_gap_aboveaxis += 1

    if underaxis < aboveaxis:
        for j in range(len(contours_list)):
            # Data
            ref_point = ref_point_list[j]
            c = contours_list[j][1]
            c_gap = np.asarray(phantom[j])
            c_mirror = np.asarray(mirrored_phantom[j])

            # Set the segments of interest
            seg_c_mirror = []
            for n in range(0, c_mirror.shape[0]):
                point = c_mirror[n]
                if point[1] < ref_point[1]:
                    seg_c_mirror.append(point)

            seg_c_gs = []
            for n in range(0, c.shape[0]):
                p = c[n]
                point = converter.convert(p[0], p[1], j)  # conversion of GS contour to patient space
                if point[1] < ref_point[1]:
                    seg_c_gs.append(point)

            seg_c_gap = []
            for n in range(0, c_gap.shape[0]):
                point = c_gap[n]
                if point[1] < ref_point[1]:
                    seg_c_gap.append(point)

            hemi_gold_standard[j] = np.asarray(seg_c_gs)
            hemi_mirrored_phantom[j] = np.asarray(seg_c_mirror)
            hemi_phantom[j] = np.asarray(seg_c_gap)
    else:
        for j in range(len(contours_list)):
            # Data
            ref_point = ref_point_list[j]
            c = contours_list[j][1]
            c_gap = np.asarray(phantom[j])
            c_mirror = np.asarray(mirrored_phantom[j])

            # Set the segments of interest
            seg_c_mirror = []
            for n in range(0, c_mirror.shape[0]):
                point = c_mirror[n]
                if point[1] > ref_point[1]:
                    seg_c_mirror.append(point)

            seg_c_gs = []
            for n in range(0, c.shape[0]):
                p = c[n]
                point = converter.convert(p[0], p[1], j)  # conversion of GS contour to patient space
                if point[1] > ref_point[1]:
                    seg_c_gs.append(point)

            seg_c_gap = []
            for n in range(0, c_gap.shape[0]):
                point = c_gap[n]
                if point[1] > ref_point[1]:
                    seg_c_gap.append(point)

            hemi_gold_standard[j] = np.asarray(seg_c_gs)
            hemi_mirrored_phantom[j] = np.asarray(seg_c_mirror)
            hemi_phantom[j] = np.asarray(seg_c_gap)

    return hemi_gold_standard, hemi_phantom, hemi_mirrored_phantom


def find_eval_region(hemi_phantom):
    """
    Determines the delimiting points of the polinomial regression evaluation region
    :param hemi_phantom: the hemisphere of interest of the mirrored phantom datasets
    :return: list of the indexes of the evaluation region initial and final points, and difference
    between the the x coordinates of these points for each contour
    """
    num_images = len(hemi_phantom)
    eval_indexes = [None] * num_images

    for i in range(num_images):
        seg_c_gap = np.asarray(hemi_phantom[i])

        # Sort points with ascending x value
        data = seg_c_gap[seg_c_gap[:, 0].argsort()]

        # Find the gap based on the greatest difference between x values in hemi phantom data
        geatest_diff = 0
        ini_gap = 0
        end_gap = 0
        for n in range(0, data.shape[0] - 1):
            diff = np.abs(data[n + 1, 0] - data[n, 0])
            if diff > geatest_diff:
                geatest_diff = diff
                ini_gap = n
                end_gap = n + 1

        region_size = 10
        ini_eval = 0
        for n in range(ini_gap, 0, -1):
            point = data[n]
            dist = np.sqrt((point[0] - data[ini_gap, 0]) ** 2 + (point[1] - data[ini_gap, 1]) ** 2)  # dist euclidiana
            if dist >= region_size:
                ini_eval = n
                break

        end_eval = 0
        for n in range(end_gap, data.shape[0]):
            point = data[n]
            dist = np.sqrt((point[0] - data[end_gap, 0]) ** 2 + (point[1] - data[end_gap, 1]) ** 2)  # dist euclidiana
            if dist >= region_size:
                end_eval = n
                break

        dif = np.abs(data[end_gap, 0] - data[ini_gap, 0])
        # eval_indexes[i] = (ini_eval, end_eval, dif, ini_gap, end_gap)
        eval_indexes[i] = (data[ini_eval, 0], data[ini_eval, 1], data[end_eval, 0], data[end_eval, 0], dif)

        # plt.plot(data[ini_eval, 0], data[ini_eval, 1], 'bx', markersize=10)
        # plt.plot(data[end_eval, 0], data[end_eval, 1], 'gx', markersize=10)
        # plt.plot(data[ini_gap, 0], data[ini_gap, 1], 'mx', markersize=10)
        # plt.plot(data[end_gap, 0], data[end_gap, 1], 'rx', markersize=10)
        # plt.plot(data[:, 0], data[:, 1], 'bo', markersize=1)
        # plt.plot(hemi_phantom[i][:, 0], hemi_phantom[i][:, 1], 'mx', markersize=1)
        # plt.xlabel('X (mm)')
        # plt.ylabel('Y (mm)')
        # plt.title(str(i))
        # plt.show()

        plt.plot(data[ini_eval, 0], data[ini_eval, 1], 'bx', markersize=10)
        plt.plot(data[end_eval, 0], data[end_eval, 1], 'gx', markersize=10)
        plt.plot(data[ini_gap, 0], data[ini_gap, 1], 'mx', markersize=10)
        plt.plot(data[end_gap, 0], data[end_gap, 1], 'rx', markersize=10)
        plt.plot(data[:, 0], data[:, 1], 'bo', markersize=1)
        plt.plot(hemi_phantom[i][:, 0], hemi_phantom[i][:, 1], 'mx', markersize=1)
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title(str(i))
        plt.show()

    return eval_indexes


def calculate_remq(squared_dist_list):
    a = np.sum(squared_dist_list)
    b = 1.0/squared_dist_list.size
    remq = np.sqrt(b*a)
    return remq


def tests(hemi_gold_standard, hemi_phantom, hemi_mirrored_phantom, eval_indexes, c, r):
    num_images = len(hemi_gold_standard)
    eval_indexes = np.asarray(eval_indexes)
    remq_a = []
    remq_b = []
    remq_c = []
    remq_d = []
    h_a = []
    h_b = []
    h_c = []
    h_d = []
    count_num_gap_img = 0

    for i in range(num_images):

        if eval_indexes[i][4] > 1:  # contours with this low difference does not have a gap // era 2

            count_num_gap_img += 1

            gs = np.asarray(hemi_gold_standard[i])
            phantom = np.asarray(hemi_phantom[i])
            m_phantom = np.asarray(hemi_mirrored_phantom[i])
            ini_eval = int(eval_indexes[i][0])
            end_eval = int(eval_indexes[i][1])

            # Finds eval region on gold standard by looking for the points with same x and y coordinates as ini_eval
            # and end_eval. Since phantom came from the gold standard, it is possible to have a perfect match.
            seg_gs_ini = 0
            seg_gs_end = 0
            for n in range(0, gs.shape[0]):
                point = gs[n]
                if point[0] == phantom[ini_eval, 0] and point[1] == phantom[ini_eval, 1]:
                    seg_gs_ini = n
                if point[0] == phantom[end_eval, 0] and point[1] == phantom[end_eval, 1]:
                    seg_gs_end = n

            # Finds eval region on mirrored phantom by looking for the points with x coordinate within 1 mm to ini_eval
            # and end_eval. It is not possible to find a perfect match.
            seg_mirror_ini = 0
            seg_mirror_end = 0
            for n in range(0, m_phantom.shape[0]):
                point = m_phantom[n]
                if phantom[ini_eval, 0] - 1 <= point[0] >= phantom[ini_eval, 0] + 1:
                    seg_mirror_ini = n
                if phantom[end_eval, 0] - 1 <= point[0] >= phantom[end_eval, 0] + 1:
                    seg_mirror_end = n

            # Cuting the segments based on the found indexes for each case.
            if seg_mirror_end > seg_mirror_ini:
                seg_m_phantom = m_phantom[seg_mirror_ini:seg_mirror_end, :]
            else:
                seg_m_phantom = m_phantom[seg_mirror_end:seg_mirror_ini, :]

            seg_phantom = phantom[ini_eval:end_eval, :]

            seg_gs = gs[seg_gs_ini:seg_gs_end, :]

            # plt.plot(phantom[ini_eval, 0], phantom[ini_eval, 1], 'bx', markersize=10)
            # plt.plot(phantom[end_eval, 0], phantom[end_eval, 1], 'gx', markersize=10)
            # # plt.plot(seg_gs[:, 0], seg_gs[:, 1], 'bo', markersize=1)
            # plt.plot(seg_m_phantom[:, 0], seg_m_phantom[:, 1], 'go', markersize=3)
            # plt.plot(seg_phantom[:, 0], seg_phantom[:, 1], 'mx', markersize=1)
            # plt.xlabel('X (mm)')
            # plt.ylabel('Y (mm)')
            # plt.title(str(i))
            # plt.show()

            # Teste A - Segmento do contorno com falha (phantom)
            # Create polynomial based on seg_phantom
            z = np.polyfit(seg_phantom[:, 0], seg_phantom[:, 1], 4)  # polinomio
            ptos = 0.2  # resolução em x
            x = np.arange(seg_phantom[0, 0], seg_phantom[len(seg_phantom)-1, 0], ptos)
            est = np.polyval(z, x)  # avaliação do polinomio em x
            pred = np.vstack((x, est)).T  # organização do valor de cada ponto do polinômio (y) no respectivo x
            # REMQ e Hausdorff
            min_dist_list = []
            for j in range(pred.shape[0]):
                pred_p = pred[j]
                min_dist = 100
                for k in range(seg_gs.shape[0]):
                    gs_p = seg_gs[k, 0:2]
                    dist = np.sqrt((pred_p[0] - gs_p[0]) ** 2 + (pred_p[1] - gs_p[1]) ** 2)
                    if dist <= min_dist:
                        min_dist = dist
                min_dist_list.append(min_dist)
            h = np.amax(np.asarray(min_dist_list))
            squared_dist_list = np.asarray(min_dist_list)**2
            remq = calculate_remq(squared_dist_list)
            remq_a.append(remq)
            h_a.append(h)

            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax2.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.plot(seg_phantom[:, 0], seg_phantom[:, 1], 'ro', markersize=1)
            # ax1.plot(seg_m_phantom[:, 0], seg_m_phantom[:, 1], 'go', markersize=1)
            # ax1.plot(seg_gs[:, 0], seg_gs[:, 1], 'ro', markersize=1)
            # ax1.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.set(xlabel='X (mm)', ylabel='Y (mm)')
            # ax1.set(xlabel='X (mm)', ylabel='Y (mm)')
            # fig.suptitle("Teste A - Image " + str(i))
            # ax1.set_aspect('equal')
            # ax2.set_aspect('equal')
            # plt.show()

            # Teste B - Segmento do contorno com falha (phantom) + segmento do contorno espelhado (m_phantom)
            # Create dataset based on seg_phantom e seg_m_phantom
            data_b = np.concatenate((seg_phantom[:, :], seg_m_phantom[:, :]), axis=0)
            # Create polynomial based on data_b
            z = np.polyfit(data_b[:, 0], data_b[:, 1], 4)  # polinomio
            ptos = 0.2  # resolução em x
            x = np.arange(seg_phantom[0, 0], seg_phantom[len(seg_phantom) - 1, 0], ptos)
            est = np.polyval(z, x)  # avaliação do polinomio em x
            pred = np.vstack((x, est)).T  # organização do valor de cada ponto do polinômio (y) no respectivo x
            # REMQ e Hausdorff
            min_dist_list = []
            for j in range(pred.shape[0]):
                pred_p = pred[j]
                min_dist = 100
                for k in range(seg_gs.shape[0]):
                    gs_p = seg_gs[k, 0:2]
                    dist = np.sqrt((pred_p[0] - gs_p[0]) ** 2 + (pred_p[1] - gs_p[1]) ** 2)
                    if dist <= min_dist:
                        min_dist = dist
                min_dist_list.append(min_dist)
            h = np.amax(np.asarray(min_dist_list))
            squared_dist_list = np.asarray(min_dist_list) ** 2
            remq = calculate_remq(squared_dist_list)
            remq_b.append(remq)
            h_b.append(h)

            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax2.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.plot(seg_phantom[:, 0], seg_phantom[:, 1], 'ro', markersize=1)
            # ax1.plot(seg_m_phantom[:, 0], seg_m_phantom[:, 1], 'go', markersize=1)
            # ax1.plot(seg_gs[:, 0], seg_gs[:, 1], 'ro', markersize=1)
            # ax1.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.set(xlabel='X (mm)', ylabel='Y (mm)')
            # ax1.set(xlabel='X (mm)', ylabel='Y (mm)')
            # fig.suptitle("Teste B - Image " + str(i))
            # ax1.set_aspect('equal')
            # ax2.set_aspect('equal')
            # plt.show()

            # Teste C - Segmento do contorno com falha (phantom) com peso 2 + segmento do contorno espelhado (m_phantom)
            # Create dataset based on seg_phantom e seg_m_phantom
            data_c = np.concatenate((seg_phantom[:, :], seg_phantom[:, :], seg_m_phantom[:, :]), axis=0)  # mudar
            # Create polynomial based on data_c
            z = np.polyfit(data_c[:, 0], data_c[:, 1], 4)  # polinomio
            ptos = 0.2  # resolução em x
            x = np.arange(seg_phantom[0, 0], seg_phantom[len(seg_phantom) - 1, 0], ptos)  # manter sempre
            est = np.polyval(z, x)  # avaliação do polinomio em x
            pred = np.vstack((x, est)).T  # organização do valor de cada ponto do polinômio (y) no respectivo x
            # REMQ e Hausdorff
            min_dist_list = []
            for j in range(pred.shape[0]):
                pred_p = pred[j]
                min_dist = 100
                for k in range(seg_gs.shape[0]):
                    gs_p = seg_gs[k, 0:2]
                    dist = np.sqrt((pred_p[0] - gs_p[0]) ** 2 + (pred_p[1] - gs_p[1]) ** 2)
                    if dist <= min_dist:
                        min_dist = dist
                min_dist_list.append(min_dist)
            h = np.amax(np.asarray(min_dist_list))
            squared_dist_list = np.asarray(min_dist_list) ** 2
            remq = calculate_remq(squared_dist_list)
            remq_c.append(remq)  # atenção mudar
            h_c.append(h)  # atenção mudar

            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax2.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.plot(seg_phantom[:, 0], seg_phantom[:, 1], 'ro', markersize=1)
            # ax1.plot(seg_m_phantom[:, 0], seg_m_phantom[:, 1], 'go', markersize=1)
            # ax1.plot(seg_gs[:, 0], seg_gs[:, 1], 'ro', markersize=1)
            # ax1.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.set(xlabel='X (mm)', ylabel='Y (mm)')
            # ax1.set(xlabel='X (mm)', ylabel='Y (mm)')
            # fig.suptitle("Teste C - Image " + str(i))
            # ax1.set_aspect('equal')
            # ax2.set_aspect('equal')
            # plt.show()

            # Teste D - Segmento do contorno com falha (phantom) + segmento do contorno espelhado (m_phantom) com peso 2
            # Create dataset based on seg_phantom e seg_m_phantom
            data_d = np.concatenate((seg_phantom[:, :], seg_m_phantom[:, :], seg_m_phantom[:, :]), axis=0)
            # Create polynomial based on data_d
            z = np.polyfit(data_d[:, 0], data_d[:, 1], 4)  # polinomio
            ptos = 0.2  # resolução em x
            x = np.arange(seg_phantom[0, 0], seg_phantom[len(seg_phantom) - 1, 0], ptos)
            est = np.polyval(z, x)  # avaliação do polinomio em x
            pred = np.vstack((x, est)).T  # organização do valor de cada ponto do polinômio (y) no respectivo x
            # REMQ e Hausdorff
            min_dist_list = []
            for j in range(pred.shape[0]):
                pred_p = pred[j]
                min_dist = 100
                for k in range(seg_gs.shape[0]):
                    gs_p = seg_gs[k, 0:2]
                    dist = np.sqrt((pred_p[0] - gs_p[0]) ** 2 + (pred_p[1] - gs_p[1]) ** 2)
                    if dist <= min_dist:
                        min_dist = dist
                min_dist_list.append(min_dist)
            h = np.amax(np.asarray(min_dist_list))
            squared_dist_list = np.asarray(min_dist_list) ** 2
            remq = calculate_remq(squared_dist_list)
            remq_d.append(remq)
            h_d.append(h)

            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax2.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.plot(seg_phantom[:, 0], seg_phantom[:, 1], 'ro', markersize=1)
            # ax1.plot(seg_m_phantom[:, 0], seg_m_phantom[:, 1], 'go', markersize=1)
            # ax1.plot(seg_gs[:, 0], seg_gs[:, 1], 'ro', markersize=1)
            # ax1.plot(pred[:, 0], pred[:, 1], 'bo', markersize=1, alpha=0.5)
            # ax2.set(xlabel='X (mm)', ylabel='Y (mm)')
            # ax1.set(xlabel='X (mm)', ylabel='Y (mm)')
            # fig.suptitle("Teste D - Image " + str(i))
            # ax1.set_aspect('equal')
            # ax2.set_aspect('equal')
            # plt.show()

    # Writes results to file

    f = open("resultsteste.txt", "a")
    f.write("Patient c and r: " + str(c) + " " + str(r) + "\n")
    f.write("Number of images with gap: " + str(count_num_gap_img) + "\n")

    remq_a_mean = round(np.mean(np.asarray(remq_a)), 3)
    remq_a_desvpad = round(np.std(np.asarray(remq_a)), 3)
    remq_b_mean = round(np.mean(np.asarray(remq_b)), 3)
    remq_b_desvpad = round(np.std(np.asarray(remq_b)), 3)
    remq_c_mean = round(np.mean(np.asarray(remq_c)),3)
    remq_c_desvpad = round(np.std(np.asarray(remq_c)), 3)
    remq_d_mean = round(np.mean(np.asarray(remq_d)), 3)
    remq_d_desvpad = round(np.std(np.asarray(remq_d)), 3)

    f.write(str(remq_a_mean) + " ± " + str(remq_a_desvpad) + "," + str(remq_b_mean) + " ± "
            + str(remq_b_desvpad) + "," + str(remq_c_mean) + " ± " + str(remq_c_desvpad) + ","
            + str(remq_d_mean) + " ± " + str(remq_d_desvpad) + "\n")

    h_a_mean = round(np.mean(np.asarray(h_a)), 3)
    h_a_desvpad = round(np.std(np.asarray(h_a)),3)
    h_b_mean = round(np.mean(np.asarray(h_b)),3)
    h_b_desvpad = round(np.std(np.asarray(h_b)), 3)
    h_c_mean = round(np.mean(np.asarray(h_c)), 3)
    h_c_desvpad = round(np.std(np.asarray(h_c)), 3)
    h_d_mean = round(np.mean(np.asarray(h_d)), 3)
    h_d_desvpad = round(np.std(np.asarray(h_d)), 3)

    f.write(str(h_a_mean) + " ± " + str(h_a_desvpad) + "," + str(h_b_mean) + " ± "
            + str(h_b_desvpad) + "," + str(h_c_mean) + " ± " + str(h_c_desvpad) + ","
            + str(h_d_mean) + " ± " + str(h_d_desvpad) + "\n")

    f.close()


def axisequal3d(ax):
    # Function found on internet to fix 3D plot aspect ratio
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def main():
    # Dataset reading and contours
    # [r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P1",
    #              r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P2",
    paths = [r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P1",
             r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P2",
             r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P3",
             r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P4",
             r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P5"]
    for n in range(len(paths)):
        datasets = load_dicom_folder(paths[n])
        # por if aqui depois pra n de 3 a 5
        # datasets = np.flip(datasets, 0)
        series_arr, converter = dicom_datasets_to_numpy(datasets)
        num_images = series_arr.shape[2]
        contours_list = [None] * num_images  # list of all contours of all images

        for i in range(num_images):
            img = series_arr[:, :, i]
            # print("Image " + str(i))
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

        # Contours_list plot
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # for b in range(num_images):
        #     contour = contours_list[b][1]  # longer external contour
        #     ax.plot(contour[:, 0], contour[:, 1], b, 'bo', markersize=1, alpha=0.5)
        # ax.set_xlim3d(0, 512)
        # ax.set_ylim3d(0, 512)
        # ax.set_zlim3d(0, num_images)
        # axisequal3d(ax)
        # ax.set_aspect('equal')
        # plt.show()

        # Create phantom
        # c = [[218, 404, 30], [283, 380, 60], [], [0,0,0], [0,0,0]]  # p1, p2, p3, p4, p5
        c = [[218, 404, 30], [283, 380, 60]]  # p1, p2
        r = [10, 20, 30]  # p, m, g

        for k in range(len(r)):
            phantom = create_phantom(contours_list, c[n], r[k], converter)
            phantom = np.asarray(phantom)
            print("Phantom done")

            # Phantom plot
            # fig2 = plt.figure()
            # ax = Axes3D(fig2)
            # for m in range(num_images):
            #     contour = np.asarray(phantom[m])
            #     ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], 'bo', markersize=1, alpha=0.5)
            # ax.set_ylabel('Y (mm)')
            # ax.set_xlabel('X (mm)')
            # ax.set_zlabel('Z (mm)')
            # axisequal3d(ax)
            # ax.set_aspect('equal')
            # plt.show()

            mirrored_phantom, ref_point_list = mirror_contours(phantom)
            print("Mirrored Phantom done")

            hemi_gold_standard, hemi_phantom, hemi_mirrored_phantom = find_hemisphere(contours_list,
                                                                                      phantom, mirrored_phantom,
                                                                                      ref_point_list, converter)
            print("Hemi datasets done")

            # fig = plt.figure()
            # ax = Axes3D(fig)
            # for o in range(num_images):
            #     contour = np.asarray(hemi_phantom[o])
            #     ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], 'bo', markersize=1, alpha=0.5)
            #     contour = np.asarray(hemi_mirrored_phantom[o])
            #     ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], 'go', markersize=1, alpha=0.5)
            #     # contour = np.asarray(hemi_gold_standard[k])
            #     # ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], 'mo', markersize=1, alpha=0.5)
            # ax.set_ylabel('Y (mm)')
            # ax.set_xlabel('X (mm)')
            # ax.set_zlabel('Z (mm)')
            # axisequal3d(ax)
            # ax.set_aspect('equal')
            # plt.show()

            eval_indexes = find_eval_region(hemi_phantom)
            print("Evaluation intervals set")

            print("Testing... P" + str(c[n]) + " Size " + str(r[k]))
            tests(hemi_gold_standard, hemi_phantom, hemi_mirrored_phantom, eval_indexes, c[n], r[k])
            print("Tests done")

    # ref_point = ref_point_list[60]  # trocar n img
    # ref_vector = np.array([1, 0])
    # p = np.asarray(phantom[60])  # trocar n img
    # plt.plot(p[:, 0], p[:, 1], 'mo', markersize=1)
    # plt.plot(mirrored_phantom[60][:, 0], mirrored_phantom[60][:, 1], 'bo', markersize=1)  # trocar n img
    # far_point = ref_point + 100 * ref_vector
    # mid_point = ref_point - 100 * ref_vector
    # plt.plot([mid_point[0], far_point[0]], [mid_point[1], far_point[1]], 'r--')
    # plt.plot(ref_point[0], ref_point[1], 'gx', markersize=10)
    # plt.xlabel('X (mm)')
    # plt.ylabel('Y (mm)')
    # plt.show()


if __name__ == '__main__':
    main()

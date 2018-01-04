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
    print("Set " + str(len(contours_wanted)) + " contours of interest")
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
    mean_points_real = [0, 0, 0]  # to points on the skull axis line (healthy slices)
    contours_list = [None] * series_arr.shape[2]  # list of all contours of all slices
    contours_mean_point_list = [None] * series_arr.shape[2]  # list of all mean points of contours of interest
    # Converts all contours for patient coordinate space based on DICOM tag information
    for i in range(series_arr.shape[2]):
        img = series_arr[:, :, i]
        img_orient_pat = [float(x) for x in list(datasets[i].ImageOrientationPatient)]
        img_position_pat = [float(x) for x in list(datasets[i].ImagePositionPatient)]
        pixel_spacing = [float(x) for x in list(datasets[i].PixelSpacing)]
        iop1 = np.array(img_orient_pat[0:3])
        iop2 = np.array(img_orient_pat[3:6])
        [cw, pma] = select_contours(img)
        cw_real = copy.copy(cw)
        for contour in cw_real:
            for k in range(len(contour)):
                contour[k] = img_position_pat \
                             + iop1 * pixel_spacing[1] * contour[k][0] \
                             + iop2 * pixel_spacing[0] * contour[k][1]
        contours_list[i] = cw_real
        # pixel_mean_array_list[i] = pma # list of mean contour points for every slice, on pixel coord system
        # collect points to skull axial axis calculation
        if len(pma) == 2:  # healthy skull slice has outside and inside contours (=2)
            # uses the mean point of only one contour, no problem because contours are approx. concentric
            pixel_mean_real = img_position_pat \
                              + iop1 * pixel_spacing[1] * pma[0][0] \
                              + iop2 * pixel_spacing[0] * pma[0][1]
            contours_mean_point_list[i] = pixel_mean_real
            mean_points_real = np.vstack([mean_points_real, pixel_mean_real])

    return contours_list, mean_points_real, contours_mean_point_list


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
    :param z: z coordinate of the wanted point 
    :return: 3D ndarray with x, y and z coordinates of the wanted point
    """
    # s*direction + point = [x, y, z]
    s = (z - point[2])/direction[2]
    x = point[0] + s * direction[0]
    y = point[1] + s * direction[1]
    return np.array([x, y, z])


def main():
    # datasets = load_dicom_folder(r"C:\Users\Escritorio\Dropbox\USP\Projeto Mariana\TestSeries\daniel\OSSOCopy")
    datasets = load_dicom_folder(r"C:\Users\Escritorio\Dropbox\USP\Projeto Mariana\TestSeries\nic")  #Nicenaldo
    # datasets = load_dicom_folder(r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\nic")
    # datasets = load_dicom_folder(r"C:\Users\Escritorio\Dropbox\USP\Projeto Mariana\TestSeries\D10A2878") #Darci
    series_arr, _ = dicom_datasets_to_numpy(datasets)

    contours_list, mean_points_real, contours_mean_point_list = \
        contours_to_patient_coord_sys_and_points_to_skull_axial_axis(datasets, series_arr)

    # Calculates direction and mean point to define skull axial axis
    mpr = copy.copy(mean_points_real[1:, :])  # first point is for inicialization only
    direction, mean = calculate_line_from_points(mpr)

    # Calculates contour mean point for bone missing skull slices using skull axial axis
    mean_points_real_gap = [0, 0, 0]  # to points on the skull axis line (bone missing slices)
    for i in range(series_arr.shape[2]):
        if len(contours_list[i]) == 1:  # bone missing skull slice has only one contour
            img_position_pat = [float(x) for x in list(datasets[i].ImagePositionPatient)]
            mean_points_real_gap = np.vstack([mean_points_real_gap, point_on_line(mean, direction, img_position_pat[2])])
            contours_mean_point_list[i] = mean_points_real_gap
    mprg = copy.copy(mean_points_real_gap[1:, :])  # first point is for inicialization only

    # Plots in blue central contour points of healthy slices (ref points for axial axis), plots in red central contour
    # points calculated for bone missing slices, plots all contours from contours_list
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(mpr[:, 0], mpr[:, 1], mpr[:, 2])
    # ax.scatter(mprg[:, 0], mprg[:, 1], mprg[:, 2], c='red')
    # for j in range(len(contours_list)):
    #     for contour in contours_list[j]:
    #         ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], linewidth=1)
    # ax.set_xlim3d(-110, 110)
    # ax.set_ylim3d(-10, 210)
    # ax.set_zlim3d(90, 310)
    # plt.axis('scaled')
    # p1 = point_on_line(mean, direction, 160)  # reference point for line plot
    # p2 = point_on_line(mean, direction, 240)  # reference point for line plot
    # plt.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])  # format: [x1, x2] [y1, y2] [z1, z2]
    # plt.show()

    # próximo passo: calcular eixo de simetria dos contornos com falha , guardar o z do slice,
    # guardar side_a e side_b_m
    # gap_contour_sides = [[None]*2]*series_arr.shape[2]  # lista com mesmo n que há de slices
    # for i in range(series_arr.shape[2]):
    #   if len(contours_list[i]) == 1:  # bone missing skull slice has only one contour
    img_position_pat = [float(x) for x in list(datasets[40].ImagePositionPatient)]
    pm = point_on_line(mean, direction, img_position_pat[2])
    pm2d = pm[:2]
    theta = define_contour_axis(pm2d, contours_list[40][0][:, :2])
    [a, b, bm] = set_mirror_contour_sides(contours_list[40][0][:, :2], pm2d, -theta)
    img = series_arr[:, :, 40]
    plot_contours(img, [a, b, bm])


def define_contour_axis(pixel_mean, contour):
    """
    
    :param pixel_mean: 
    :param contour: 
    :return: 
    """
    # pixel_mean = np.mean(contour, axis=0)  # mean point of the contour
    theta_std = [0, 0]  # contains line "angle" and respective mean deviation between contour sides

    for theta in range(-20, 21, 1):  # last range value is not included (ini, end-1, delta)
        ref_vector = np.array([1, 0])
        theta_rad = np.deg2rad(theta)
        rot_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                               [np.sin(theta_rad), np.cos(theta_rad)]])
        ref_vector = np.matmul(ref_vector, rot_matrix)
        ref_vector = ref_vector / np.sqrt(ref_vector.dot(ref_vector))  # vector normalization

        # Create 2 sides of the contour with reference to line (ref_vector, pixel_mean)
        side_a, side_b = classify_contour_point(contour, pixel_mean, ref_vector)

        # Identify the side with the bone gap, which is the one with less contour points
        # The side with gap is the one to be mirrored, so it has to be always side_b
        if side_a.shape[0] < side_b.shape[0]:
            side_a, side_b = side_b, side_a

        # Mirror side b with respect to line (ref_vector, pixel_mean)
        side_b_m = mirror_contour_point(pixel_mean, ref_vector, side_b)

        # Measure similarity between points on side_a and side_b_m
        min_dist = np.zeros([side_a.shape[0]], dtype=np.float64)  # auxiliary vector to contain distance between points
        print(theta)
        print(ref_vector)
        for i, point_a in enumerate(side_a):
            m = 100000.0
            for point_b in side_b_m:
                d = (point_b[1] - point_a[1]) ** 2 + (point_b[0] - point_a[0]) ** 2
                if d < m:
                    m = d
            min_dist[i] = np.sqrt(m)
        dv = np.std(min_dist)
        theta_std = np.vstack([theta_std, [theta, dv]])

    print("theta_std = " + str(theta_std))

    theta_final = min(theta_std[1:, :], key=lambda t: t[1])[0]

    print("theta final = " + str(theta_final))

    return theta_final


def set_mirror_contour_sides(contour, pixel_mean, theta_final):
    ref_vector = np.array([1, 0])
    theta_final_rad = np.deg2rad(theta_final)
    rot_matrix = np.array([[np.cos(theta_final_rad), -np.sin(theta_final_rad)],
                           [np.sin(theta_final_rad), np.cos(theta_final_rad)]])
    ref_vector = np.matmul(ref_vector, rot_matrix)
    ref_vector = ref_vector / np.sqrt(ref_vector.dot(ref_vector))  # vector normalization
    # Create 2 sides of the contour with reference to line (ref_vector, pixel_mean)
    side_a, side_b = classify_contour_point(contour, pixel_mean, ref_vector)
    # Identify the side with the bone gap, which is the one with less contour points
    # The side with gap is the one to be mirrored, so it has to be always side_b
    if side_a.shape[0] < side_b.shape[0]:
        side_a, side_b = side_b, side_a
    # Mirror side b with respect to line (ref_vector, pixel_mean)
    side_b_m = mirror_contour_point(pixel_mean, ref_vector, side_b)
    return side_a, side_b, side_b_m


def classify_contour_point(contour, pixel_mean, ref_vector):
    side_a = [0, 0]
    side_b = [0, 0]
    v = np.array([0, 0, 0], dtype=np.float64)  # auxiliar vector
    r = np.array([0, 0, 0], dtype=np.float64)  # auxiliar vector
    r[0:2] = ref_vector
    for point in contour:
        # Cross product between ref_vector and vector defined by pixel_mean and each contour point
        # Order the points based on the signal of 3rd component of the resultant
        v[0:2] = pixel_mean - point
        signal = np.cross(v, r)
        if signal[2] > 0:
            side_a = np.vstack([side_a, point])  # positive, right side of reference line
        else:
            side_b = np.vstack([side_b, point])  # negative and equal zero, left side of reference line
    return side_a[1:, :], side_b[1:, :]


def mirror_contour_point(pixel_mean, ref_vector, side_b):
    side_b_m = [0, 0]
    for pi in side_b[1:, :]:
        pr = pixel_mean - (pi - pixel_mean) + 2 * ref_vector * np.dot((pi - pixel_mean), ref_vector)
        side_b_m = np.vstack([side_b_m, pr])
    return side_b_m[1:, :]


def plot_contours(img, contours):
    # Display the image and plot all contours in a array of contours
    fig, ax = plt.subplots()
    contour_img = ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray, origin='bottom')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)  # x and y are switched because for correct image plot
    ax.axis('image')
    plt.colorbar(contour_img, ax=ax)
    plt.show()


if __name__ == '__main__':
    main()

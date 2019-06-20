# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import pydicom as dicom
from pydicom.errors import InvalidDicomError
import os
import numpy as np


def load_dicom_folder(folder_path):
    dcm_files = []

    print("Loading files in \"" + folder_path + "\"")

    # Get all filenames in the target folder
    for dir_name, subdir_list, file_list in os.walk(folder_path):
        for file_name in file_list:
            dcm_files.append(os.path.join(dir_name, file_name))

    # print("Found " + str(len(dcm_files)) + " files")

    # Clear non-dicom files
    datasets = []
    for file in dcm_files:
        try:
            datasets.append(dicom.read_file(file))
        except InvalidDicomError:
            print("File \"" + file + "\" is not a valid dicom file!")
            dcm_files.remove(file)

    print("Found " + str(len(dcm_files)) + " DICOM files")

    # Try to sort based on instance number then SOPInstanceUID
    try:
        datasets.sort(key=lambda x: x.InstanceNumber)
        # print("Sorted based on InstanceNumber")
    except AttributeError:
        try:
            datasets.sort(key=lambda x: x.SOPInstanceUID)
            print("Sorted based on SOPInstanceUID")
        except AttributeError:
            print("Sorted based on filenames")

    return datasets


class PatientSpaceConversion:
    def __init__(self):
        self.ipps = []
        self.iops = []
        self.pss = []

    def convert(self, x, y, z):
        ipp = self.ipps[z]
        iop = self.iops[z]
        ps = self.pss[z]

        x_patient = ipp[0] + iop[0][0]*x*ps[1] + iop[1][0]*y*ps[0]
        y_patient = ipp[1] + iop[0][1]*x*ps[1] + iop[1][1]*y*ps[0]
        z_patient = ipp[2] + iop[0][2]*x*ps[1] + iop[1][2]*y*ps[0]

        return x_patient, y_patient, z_patient


def dicom_datasets_to_numpy(datasets):
    img_dims = (int(datasets[0].Rows), int(datasets[0].Columns), len(datasets))
    # print("Series dims: " + str(img_dims))

    converter = PatientSpaceConversion()
    for i in range(len(datasets)):
        converter.pss.append([float(datasets[i].PixelSpacing[0]), float(datasets[i].PixelSpacing[1])])

        iop_string = datasets[i].ImageOrientationPatient
        iop = [float(j) for j in iop_string]
        iop_x = iop[0:3]
        norm = np.sqrt(iop_x[0]**2 + iop_x[1]**2 + iop_x[2]**2)
        iop_x = [iop_x[0]/norm, iop_x[1]/norm, iop_x[2]/norm]
        iop_y = iop[3:6]
        norm = np.sqrt(iop_y[0] ** 2 + iop_y[1] ** 2 + iop_y[2] ** 2)
        iop_y = [iop_y[0] / norm, iop_y[1] / norm, iop_y[2] / norm]
        iop = [iop_x, iop_y]
        converter.iops.append(iop)

        ipp_string = datasets[i].ImagePositionPatient
        ipp = [float(k) for k in ipp_string]
        converter.ipps.append(ipp)

    # Load pixel data into an int32 array so we never have to care about signs
    series_arr = np.zeros(img_dims, dtype='int32')
    for i, d in enumerate(datasets):
        # Also performs rescaling. 'unsafe' since it converts from float64 to int32
        np.copyto(series_arr[:, :, i], np.flipud(d.RescaleSlope * d.pixel_array + d.RescaleIntercept), 'unsafe')

    return series_arr, converter


def dicom_datasets_to_numpy_basic(datasets):
    img_dims = (int(datasets[0].Rows), int(datasets[0].Columns), len(datasets))
    img_spacings = (
    float(datasets[0].PixelSpacing[0]), float(datasets[0].PixelSpacing[1]), float(datasets[0].SliceThickness))

    print("Series dims: " + str(img_dims))
    print("Series spacings: " + str(img_spacings))
    print("Image orientation patient: " + str(datasets[0].ImageOrientationPatient))

    # Create axes
    x = np.arange(0.0, (img_dims[0] + 1) * img_spacings[0], img_spacings[0])
    y = np.arange(0.0, (img_dims[1] + 1) * img_spacings[1], img_spacings[1])
    z = np.arange(0.0, (img_dims[2] + 1) * img_spacings[2], img_spacings[2])

    # Load pixel data into an int32 array so we never have to care about signs
    series_arr = np.zeros(img_dims, dtype='int32')
    for i, d in enumerate(datasets):
        # Also performs rescaling. 'unsafe' since it converts from float64 to int32
        np.copyto(series_arr[:, :, i], np.flipud(d.RescaleSlope * d.pixel_array + d.RescaleIntercept), 'unsafe')

    return series_arr, (x, y, z)


# def main():
    # Tests
    # x = 1
    # y = 2
    # z = 0
    #
    # ipp = [1, 1, 1]
    # iop = [1, 0, 0, 0, 0.7071, 0.7071]
    # ps = [0.3, 0.4]
    #
    # x_patient = ipp[0] + iop[0] * x * ps[1] + iop[3] * y * ps[0]
    # y_patient = ipp[1] + iop[1] * x * ps[1] + iop[4] * y * ps[0]
    # z_patient = ipp[2] + iop[2] * x * ps[1] + iop[5] * y * ps[0]
    #
    # print(str(x_patient))
    # print(str(y_patient))
    # print(str(z_patient))

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # for k in range(3):
    #     img = arr[k]
    #     ax.plot(img[:, 0], img[:, 1], img[0][2], 'b', markersize=5, alpha=0.5)
    # ax.set_xlim3d(0, 5)
    # ax.set_ylim3d(0, 5)
    # ax.set_zlim3d(0, 11)
    # plt.axis('scaled')
    # plt.show()

#
# if __name__ == '__main__':
#    main()
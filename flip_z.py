import glob
import numpy as np
import re
import pydicom as dicom


def main():
    orig_folder_path = r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P5\*.dcm"
    dest_folder_path = r"C:\Users\Mariana\Dropbox\USP\Projeto Mariana\TestSeries\P5reorg"

    orig_names = glob.glob(orig_folder_path)
    num_images = len(orig_names)

    for name in orig_names:
        # Get just filename without the extension or the rest of the folder path
        name_no_ext = re.findall(r"([^\\-]+)\.dcm", name)[0]
        im_number = int(name_no_ext)
        new_im_number = num_images - (im_number - 14) + 1

        orig_file = dicom.read_file(name)
        uid = re.findall(r"(.*?)[^.]*$", str(orig_file.SOPInstanceUID))[0]

        # Change image info
        orig_file.ImageNumber = new_im_number
        orig_file.InstanceNumber = new_im_number

        # Save under a new filename
        full_dest_path = dest_folder_path + "\\" + str(new_im_number).zfill(5) + ".dcm"
        orig_file.save_as(full_dest_path)
        print("edited and saved to " + str(full_dest_path))


if __name__ == '__main__':
    main()
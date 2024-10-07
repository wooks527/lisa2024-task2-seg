import argparse
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_closing
from natsort import natsorted
from tqdm import tqdm


def process_case(case, img_path, out_path):
    img_subpath = os.path.join(img_path, case)
    out_subpath = os.path.join(out_path, case)

    img = sitk.ReadImage(img_subpath)

    orig_spacing = img.GetSpacing()
    orig_origin = img.GetOrigin()
    orig_direction = img.GetDirection()
    img_arr = sitk.GetArrayFromImage(img)

    labels_to_process = [1, 2]

    processed_img_arr = np.copy(img_arr)

    for label in labels_to_process:
        mask = img_arr == label

        closed_mask = binary_closing(mask, iterations=3)

        processed_img_arr[mask] = 0
        processed_img_arr[closed_mask] = label

    processed_img_arr[(processed_img_arr == 3) | (processed_img_arr == 4)] = 0

    processed_img = sitk.GetImageFromArray(processed_img_arr)
    processed_img.SetSpacing(orig_spacing)
    processed_img.SetOrigin(orig_origin)
    processed_img.SetDirection(orig_direction)

    sitk.WriteImage(processed_img, out_subpath)


def main(img_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cases = natsorted([f for f in os.listdir(img_path) if f.endswith(".nii.gz")])

    for case in tqdm(cases):
        process_case(case, img_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Input directory containing the .nii.gz images",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Output directory where the processed images will be saved",
    )
    args = parser.parse_args()

    main(args.img_path, args.out_path)

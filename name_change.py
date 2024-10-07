import argparse
import os
import shutil

from glob import glob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        help="prediction directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pred_paths = sorted(glob(f"{args.pred_dir}/*.nii.gz"))

    for pred_path in pred_paths:
        filename = os.path.basename(pred_path)
        filename = filename.replace("LISA_", "LISA_TESTING_SEG_0")
        filename = filename.replace(".nii.gz", "_CISO.nii.gz")
        filename2 = f"{args.output_dir}/{filename}"
        shutil.copyfile(pred_path, filename2)

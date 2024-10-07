import argparse
import os
import shutil
import zipfile

from glob import glob


def zip_files(file_paths, output_zip_path):
    with zipfile.ZipFile(output_zip_path, "w") as zipf:
        for file in file_paths:
            zipf.write(file, os.path.basename(file))


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
    shutil.rmtree(f"{args.output_dir}/LISA_LF_SEG_predictions", ignore_errors=True)
    os.makedirs(f"{args.output_dir}/LISA_LF_SEG_predictions")

    pred_paths = sorted(glob(f"{args.pred_dir}/*.nii.gz"))
    sub_paths = []
    for pred_path in pred_paths:
        filename = os.path.basename(pred_path)
        filename = filename.replace("LISA", "LISA_HF")
        filename = filename.replace(".nii.gz", "_hipp_prediction.nii.gz")
        sub_path = f"{args.output_dir}/LISA_LF_SEG_predictions/{filename}"
        shutil.copyfile(pred_path, sub_path)
        sub_paths.append(sub_path)

    output_zip_path = f"{args.output_dir}/LISA_LF_SEG_predictions.zip"
    zip_files(sub_paths, output_zip_path)

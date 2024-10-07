# Task 2: Segmentation

## Task Description

Participants are tasked with developing deep learning methods for automatic segmentation of the bilateral hippocampi in ultra-low-field (0.064T) T2-weighted MRI images of early childhood brains. The hippocampi are vital for cognitive and memory functions, making accurate segmentation crucial for understanding abnormal neurodevelopment.

## Data Structure

The data is organized as follows:

```
workspace
├──nnUNet_raw
│  ├──imagesTr
│  ├──imagesTs
│  └──labelsTr
├──nnUNet_preprocessed
└──nnUNet_results
```

## Codes

### Docker Environment

All code has been tested and runs within a Docker container.
To build and run the container, use the following commands:

```
docker build . -t wooks527/lisa2024_task2_seg
```

```
docker run -itd \
  --name lisa2024_task2_seg \
  -v [DATA_DIR]:/workspace/nnUNet_raw/Dataset901_LISA \
  -v [PROJECT_DIR]:/workspace \
  --device=nvidia.com/gpu=all \
  --shm-size=320g \
  wooks527/lisa2024_task2_seg:latest
```
- `DATA_DIR`: Path to the directory where the data is located.
- `PROJECT_DIR`: Path to the directory where the code is stored.


### Additional Installation

To set up the nnUNet environment, follow these steps:

```
pip install -e .
```

Set environment variables for nnUNet:

```
export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"
```

### Preprocessing

To preprocess the dataset:

```
python3 lisa_preprocessing.py
nnUNetv2_plan_and_preprocess -d 901 --verify_dataset_integrity -c 3d_fullres
```

### Training

To train the model:

```
nnUNetv2_train 901 3d_fullres FOLD -tr nnUNetTrainer_500epochs_NoMirroring
```

### Prediction

To generate predictions:

```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d 901 -c 3d_fullres -tr  nnUNetTrainer_500epochs_NoMirroring -f FOLD -chk checkpoint_latest.pth --save_probabilities
```

### Ensemble Prediction

To perform ensemble predictions:

```
nnUNetv2_ensemble -i OUTPUT_FOLDER1 OUTPUT_FOLDER2 OUTPUT_FOLDER3 OUTPUT_FOLDER4 OUTPUT_FOLDER5 -o OUTPUT_ENSEMBLE_FOLDER
```

### Postprocessing

To apply postprocessing to the ensemble results:

```
python3 lisa_postprocessing.py --img_path OUTPUT_ENSEMBLE_FOLDER --out_path OUTPUT_ENSEMBLE_POST_FOLDER
```

## References

- https://www.synapse.org/Synapse:syn55249552/wiki/626951
- https://github.com/LISA2024Challenge/Task2

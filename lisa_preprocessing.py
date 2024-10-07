import os
from natsort import natsorted

path_image = "/workspace/data/Testing_data"

images = sorted(os.listdir(path_image))
images = natsorted(images)

for idx, i in enumerate(images):
    name = i.split(".")[0]
    new_name = "LISA_" + str(idx + 1).zfill(3) + "_0000.nii.gz"
    os.rename(os.path.join(path_image, i), os.path.join(path_image, new_name))

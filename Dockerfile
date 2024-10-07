FROM docker.io/pytorch/pytorch:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONDONTWRITEBYTECODE=1

RUN sed -i 's|http://[a-z]\+.ubuntu.com|https://mirror.kakao.com|g' /etc/apt/sources.list
RUN printf "%s\n"\
  "[global]"\
  "index-url=https://mirror.kakao.com/pypi/simple/"\
  "extra-index-url=https://pypi.org/simple/"\
  "trusted-host=mirror.kakao.com"\
  > /etc/pip.conf && pip install --no-cache-dir -U pip && pip install --no-cache-dir jupyter

RUN apt-get update -qq && apt-get install -qqy\
  sudo\
  tzdata\
  vim\
  tmux\
  curl\
  jq\
  git\
  libgl1-mesa-glx\
  libglib2.0-0\
  ffmpeg x264 libx264-dev
RUN apt-get update -y
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install -y openslide-tools
RUN apt-get install python3-pip -y
RUN pip install opencv-python albumentations numpy SimpleITK nibabel pydicom==2.1.1
RUN pip install Pillow scipy matplotlib tqdm natsort seaborn easydict
RUN pip install scikit-learn h5py monai scikit-learn ffmpeg-python
RUN pip install --upgrade pip setuptools wheel
RUN pip install tensorboard monai wandb
RUN pip install numpy==1.26.4 pandas scikit-image pickle5 openpyxl
RUN pip install click requests pyspng ninja nnunet nnunetv2
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --trusted-host download.pytorch.org -r /tmp/requirements.txt
RUN apt clean && rm -rf /var/lib/apt/lists/*

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    tzdata sudo vim less curl jq git ca-certificates apt-transport-https gnupg \
    wget software-properties-common apt-utils xz-utils build-essential

RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-dev python3-virtualenv python3-opencv

RUN useradd -ms /bin/bash -d /home/ubuntu -G sudo ubuntu
RUN echo "ubuntu:123" | chpasswd

USER ubuntu
WORKDIR /home/ubuntu

ENV VIRTUAL_ENV=/home/ubuntu/venv
RUN python3.6 -m virtualenv --python=/usr/bin/python3.6 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# https://github.com/pytorch/pytorch/issues/20477
# 1.14 doesnt update
RUN pip install tensorboard==1.13.1

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /home/ubuntu/.cache/torch/checkpoints/vgg16-397923af.pth
RUN mkdir -p /home/ubuntu/.cache/torch/checkpoints/
RUN wget -O /home/ubuntu/.cache/torch/checkpoints/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth

COPY --chown=ubuntu:ubuntu *py ./
COPY --chown=ubuntu:ubuntu util/*py util/

ENV CUDA_HOME="/usr/local/cuda/"

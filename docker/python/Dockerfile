# FROM nvidia/cuda:12.3.1-base-ubuntu20.04
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime


RUN apt update
RUN apt install -y python3 python3-pip
RUN pip3 install ultralytics
RUN pip3 install more_itertools
# RUN pip3 install -y build-essential libssl-dev libffi-dev python3-dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt install ffmpeg libsm6 libxext6  -y
RUN pip3 install mysql-connector-python
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /python/yolo

# ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

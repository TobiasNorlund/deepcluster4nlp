#FROM ubuntu:14.04
#FROM anibali/pytorch:cuda-8.0
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y \
    python2.7 \
    python-pip \
    git \
    vim \
    wget \
    curl \
    cmake \
    build-essential

RUN curl -o miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh &&\
    chmod +x miniconda.sh &&\
    ./miniconda.sh -b -p /opt/conda &&\
    rm miniconda.sh

RUN export PATH=$PATH:/opt/conda/bin &&\
    conda create -n pytorch36 python=3.6

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "pytorch36", "/bin/bash", "-c"]
ENV PATH="/opt/conda/bin:${PATH}"

RUN export CMAKE_PREFIX_PATH=/opt/conda/ &&\
    conda install numpy mkl setuptools cmake cffi scikit-learn &&\
    apt-get -y install gcc libblas-dev liblapack-dev &&\
    conda install -c soumith magma-cuda80 &&\
    pip install torchvision==0.1.8

RUN pip install torch==0.4.1 \
                pytorch-nlp \
                transformers \
                faiss-cpu \
                bpython \
                wandb

RUn conda init bash

#  RUN pip install bpython future
RUN echo "conda activate pytorch36" >> /root/.bashrc

ENV PYTHONPATH="/usr/src/app/faiss-1.3.0/python/:${PYTHONPATH}"
COPY . .

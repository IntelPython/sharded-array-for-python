# Arrange order to minimize rebuilds. Things that change more often
#  are last
FROM ubuntu:20.04

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends g++ gcc wget make git\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MINICONDA=Miniconda3-latest-Linux-x86_64.sh
COPY conda-env.yml /tmp/conda-env.yml
RUN wget -q --no-check-certificate https://repo.anaconda.com/miniconda/$MINICONDA \
    && chmod a+rwx $MINICONDA \
    && ./$MINICONDA -b \
    &&  rm ./$MINICONDA \
    && /root/miniconda3/bin/conda update -y -n base -c defaults conda \
    && /root/miniconda3/bin/conda env create -n ddpt -f /tmp/conda-env.yml \
    && /root/miniconda3/bin/conda clean -y -a \
    && /root/miniconda3/bin/conda init bash

ENV PATH=${PATH}:/root/miniconda3/bin

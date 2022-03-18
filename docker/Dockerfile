# Arrange order to minimize rebuilds. Things that change more often
#  are last
FROM ubuntu:20.04

# Proxies for working inside firewall
ENV no_proxy=.intel.com
ENV https_proxy=http://proxy-chain.intel.com:912
ENV http_proxy=http://proxy-chain.intel.com:911
ENV ftp_proxy=http://proxy-chain.intel.com:911

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get install -y --no-install-recommends g++ gcc wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MINICONDA=Miniconda3-latest-Linux-x86_64.sh
RUN wget -q --no-check-certificate https://repo.anaconda.com/miniconda/$MINICONDA \
    && chmod a+rwx $MINICONDA \
    && ./$MINICONDA -b \
    &&  rm ./$MINICONDA \
    && /root/miniconda3/bin/conda update -n base -c defaults conda

ENV PATH=${PATH}:/root/miniconda3/bin

SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["/bin/bash", "--login", "-c"]
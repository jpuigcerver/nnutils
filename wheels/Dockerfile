FROM ubuntu:16.04
LABEL maintainer "Joan Puigcerver <joapuipe@gmail.com>"
LABEL com.nvidia.volumes.needed="nvidia_driver"

# Essential stuff
RUN apt-get update --fix-missing && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      git \
      libssl-dev \
      software-properties-common

# Python repositories
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 2.7
RUN apt-get install -y --no-install-recommends \
    libpython2.7 \
    libpython2.7-dev \
    libpython2.7-minimal \
    libpython2.7-stdlib \
    python2.7-minimal

# Install Python 3.5
RUN apt-get install -y --no-install-recommends \
      libpython3.5 \
      libpython3.5-dev \
      libpython3.5-minimal \
      libpython3.5-stdlib \
      python3.5-minimal

# Install Python 3.6
RUN apt-get install -y --no-install-recommends \
      libpython3.6 \
      libpython3.6-dev \
      libpython3.6-minimal \
      libpython3.6-stdlib \
      python3.6-minimal

# Install virtualenv and pip
RUN apt-get install -y --no-install-recommends \
    python-pip \
    python3-pip \
    virtualenv

# Install PyTorch for CPU-only in different Python versions.
COPY docker_pytorch_cpu.sh /
RUN /docker_pytorch_cpu.sh && \
    rm /docker_pytorch_cpu.sh

# Copy script to create wheels for CPU-only distributions.
COPY create_wheels_cpu.sh /

ENV DOCKER=1

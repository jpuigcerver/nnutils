ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL maintainer "Joan Puigcerver <joapuipe@gmail.com>"
ENV DOCKER=1

# Essential stuff
RUN apt-get update --fix-missing && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
      wget \
      libssl-dev \
      software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update

# Install Python
RUN \
for py in 2.7 3.5 3.6 3.7; do \
  apt-get install -y --no-install-recommends \
    libpython$py \
    libpython$py-dev \
    libpython$py-minimal \
    libpython$py-stdlib \
    python$py-minimal; \
done; \
apt-get install -y --no-install-recommends python3.7-distutils;

# Install pip and virtualenv
RUN \
for py in python3.6; do \
  wget https://bootstrap.pypa.io/ez_setup.py -O - | $py && \
  easy_install pip && \
  $py -m pip install -U pip && \
  $py -m pip install virtualenv; \
done;

#!/bin/bash
set -e;

SDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";
SOURCE_DIR=$(cd $SDIR/.. && pwd);

###########################################
## THIS CODE IS EXECUTED WITHIN THE HOST ##
###########################################

if [ ! -f /.dockerenv ]; then
  DOCKER_IMAGES=(
#    soumith/manylinux-cuda80
#    soumith/manylinux-cuda90
    soumith/manylinux-cuda92
  );
  for image in "${DOCKER_IMAGES[@]}"; do
    docker run --runtime=nvidia --rm --log-driver none \
	   -v /tmp:/host/tmp \
	   -v ${SOURCE_DIR}:/host/src \
	   "$image" \
	   /host/src/wheels/create_wheels_cuda.sh;
  done;
  exit 0;
fi;

#######################################################
## THIS CODE IS EXECUTED WITHIN THE DOCKER CONTAINER ##
#######################################################
set -ex;

# Install zip, apparently is not installed.
yum install -y zip openssl;

# Copy host source directory, to avoid changes in the host.
cp -r /host/src /tmp/src;
cd /tmp/src;

export PYTHON_VERSIONS=(
  cp27-cp27mu
  cp35-cp35m
  cp36-cp36m
  cp37-cp37m
);

# Detect CUDA version
export CUDA_VERSION=$(nvcc --version|tail -n1|cut -f5 -d" "|cut -f1 -d",");
export CUDA_VERSION_S="cu$(echo $CUDA_VERSION | tr -d .)";
echo "CUDA $CUDA_VERSION Detected";

if [[ "$CUDA_VERSION" == "8.0" ]]; then
  export CUDA_ARCH_LIST="3.5;5.0+PTX;5.2;6.0";
elif [[ "$CUDA_VERSION" == "9.0" ]]; then
  export CUDA_ARCH_LIST="3.5;5.0+PTX;5.2;6.0;7.0";
elif [[ "$CUDA_VERSION" == "9.2" ]]; then
  export CUDA_ARCH_LIST="3.5;5.0+PTX;5.2;6.0;6.1;7.0";
else
  exit 1;
fi;

# Install PyTorch
./wheels/install_pytorch_cuda.sh "${PYTHON_VERSIONS[@]}";

cd /tmp/src/pytorch;
for py in "${PYTHON_VERSIONS[@]}"; do
  echo "=== Building wheel for $py with CUDA ${CUDA_VERSION} ===";
  export PYTHON=/opt/python/$py/bin/python;
  $PYTHON setup.py clean;
  $PYTHON setup.py bdist_wheel;
done;

echo "=== Fixing wheels with CUDA ${CUDA_VERSION} ===";
../wheels/fix_deps.sh \
  dist nnutils_pytorch \
  "libcudart.so.${CUDA_VERSION}" \
  "/usr/local/cuda-${CUDA_VERSION}/lib64/libcudart.so.${CUDA_VERSION}";

rm -rf /opt/rh /usr/local/cuda*;
for py in "${PYTHON_VERSIONS[@]}"; do
  echo "=== Testing wheel for $py with CUDA ${CUDA_VERSION} ===";
  export PYTHON=/opt/python/$py/bin/python;
  cd /tmp;
  $PYTHON -m pip uninstall -y nnutils_pytorch;
  $PYTHON -m pip install nnutils_pytorch --no-index -f /tmp/src/pytorch/dist --no-dependencies -v;
  $PYTHON -m unittest nnutils_pytorch.mask_image_from_size_test;
  $PYTHON -m unittest nnutils_pytorch.adaptive_avgpool_2d_test;
  $PYTHON -m unittest nnutils_pytorch.adaptive_maxpool_2d_test;
  cd - 2>&1 > /dev/null;
done;

set +x;
ODIR="/host/tmp/nnutils_pytorch/whl/${CUDA_VERSION_S}";
mkdir -p "$ODIR";
cp /tmp/src/pytorch/dist/*.whl "$ODIR/";
echo "================================================================";
printf "=== %-56s ===\n" "Copied wheels to ${ODIR:5}";
echo "================================================================";

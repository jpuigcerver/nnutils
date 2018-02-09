IF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND AND
  CUDA_FOUND AND WITH_CUDA)
  INCLUDE_DIRECTORIES(
     "${PYTORCH_INCLUDE_DIR}"
     "${PYTORCH_INCLUDE_DIR}/TH"
     "${PYTORCH_INCLUDE_DIR}/THC")
  CUDA_ADD_LIBRARY(pytorch_gpu_ STATIC
    ${PROJECT_SOURCE_DIR}/nnutils/gpu/mask_image_from_size.h
    ${PROJECT_SOURCE_DIR}/torch/mask_image_from_size.h
    mask_image_from_size.h
    mask_image_from_size.cu

    ${PROJECT_SOURCE_DIR}/nnutils/gpu/adaptive_avgpool_2d.h
    ${PROJECT_SOURCE_DIR}/torch/adaptive_avgpool_2d.h
    adaptive_avgpool_2d.h
    adaptive_avgpool_2d.cu

    ${PROJECT_SOURCE_DIR}/nnutils/gpu/adaptive_maxpool_2d.h
    ${PROJECT_SOURCE_DIR}/torch/adaptive_maxpool_2d.h
    adaptive_maxpool_2d.h
    adaptive_maxpool_2d.cu)
  TARGET_LINK_LIBRARIES(pytorch_gpu_
    ${PYTORCH_TH_LIBRARY}
    ${PYTORCH_THC_LIBRARY}
    ${CUDA_LIBRARIES}
    m)
ENDIF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND AND
  CUDA_FOUND AND WITH_CUDA)

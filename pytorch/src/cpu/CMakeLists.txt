IF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)
  INCLUDE_DIRECTORIES(
     "${PYTORCH_INCLUDE_DIR}"
     "${PYTORCH_INCLUDE_DIR}/TH")
  ADD_LIBRARY(pytorch_cpu_ STATIC
    ${PROJECT_SOURCE_DIR}/nnutils/cpu/mask_image_from_size.h
    ${PROJECT_SOURCE_DIR}/torch/mask_image_from_size.h
    mask_image_from_size.h
    mask_image_from_size.cc

    ${PROJECT_SOURCE_DIR}/nnutils/cpu/adaptive_avgpool_2d.h
    ${PROJECT_SOURCE_DIR}/torch/adaptive_avgpool_2d.h
    adaptive_avgpool_2d.h
    adaptive_avgpool_2d.cc

    ${PROJECT_SOURCE_DIR}/nnutils/cpu/adaptive_maxpool_2d.h
    ${PROJECT_SOURCE_DIR}/torch/adaptive_maxpool_2d.h
    adaptive_maxpool_2d.h
    adaptive_maxpool_2d.cc)
  TARGET_LINK_LIBRARIES(pytorch_cpu_
    ${PYTORCH_TH_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT})
ENDIF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)

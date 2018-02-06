IF(PYTORCH_FOUND AND WITH_PYTORCH)
  SET(_EXTERNAL_LIBS)
  SET(_EXTERNAL_DIRS)

  IF(CUDA_FOUND AND WITH_CUDA)
    CUDA_ADD_LIBRARY(nnutils_ STATIC
      ../nnutils/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.cc

      ../nnutils/gpu/mask_image_from_size.h
      src/gpu/mask_image_from_size.h
      src/gpu/mask_image_from_size.cu)
    TARGET_INCLUDE_DIRECTORIES(nnutils_
      PRIVATE "${PYTORCH_INCLUDE_DIR}"
      PRIVATE "${PYTORCH_INCLUDE_DIR}/TH"
      PRIVATE "${PYTORCH_INCLUDE_DIR}/THC")
  ELSE()
    ADD_LIBRARY(nnutils_ STATIC
      ../nnutils/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.cc)
    TARGET_INCLUDE_DIRECTORIES(nnutils_
      PRIVATE "${PYTORCH_INCLUDE_DIR}"
      PRIVATE "${PYTORCH_INCLUDE_DIR}/TH")
    LIST(APPEND _EXTERNAL_LIBS ${PYTORCH_TH_LIBRARY})
    LIST(APPEND _EXTERNAL_LIBS ${BLAS_LIBRARIES})
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # Dependency with the wrapper code.
  LIST(APPEND _EXTERNAL_LIBS ${CMAKE_CURRENT_BINARY_DIR}/libnnutils_.a)

  # Dependency with CUDA
  IF(CUDA_FOUND AND WITH_CUDA)
    LIST(APPEND _EXTERNAL_LIBS ${PYTORCH_TH_LIBRARY})
    LIST(APPEND _EXTERNAL_LIBS ${PYTORCH_THC_LIBRARY})
    LIST(APPEND _EXTERNAL_LIBS ${CUDA_LIBRARIES})
  ENDIF(CUDA_FOUND AND WITH_CUDA)

  # Dependency with threading library
  LIST(APPEND _EXTERNAL_LIBS ${CMAKE_THREAD_LIBS_INIT})

  # Add include directories needed for the Python extension compilation.
  GET_PROPERTY(_EXTERNAL_DIRS TARGET nnutils_ PROPERTY INCLUDE_DIRECTORIES)

  SET(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  SET(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  SET(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")


  CONFIGURE_FILE("${SETUP_PY_IN}"  "${SETUP_PY}")
  ADD_CUSTOM_COMMAND(OUTPUT "${OUTPUT}"
    COMMAND ${PYTHON_EXECUTABLE} "${SETUP_PY}" build
    COMMAND ${CMAKE_COMMAND} -E touch "${OUTPUT}"
    DEPENDS nnutils_pytorch/__init__.py)
  ADD_CUSTOM_TARGET(nnutils_pytorch ALL DEPENDS "${OUTPUT}" nnutils_)

  IF(NOT PYTHON_SETUP_PREFIX)
    SET(PYTHON_SETUP_PREFIX "${CMAKE_INSTALL_PREFIX}")
  ENDIF()
  MESSAGE(STATUS ${PYTHON_SETUP_PREFIX})

  IF(PYTHON_SETUP_PREFIX)
    INSTALL(
      CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=${PYTHON_SETUP_PREFIX})")
  ELSE()
    INSTALL(
      CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
  ENDIF(PYTHON_SETUP_PREFIX)
ENDIF(PYTORCH_FOUND AND WITH_PYTORCH)

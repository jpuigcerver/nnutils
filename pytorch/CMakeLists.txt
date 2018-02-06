SET(PYTORCH_SETUP_PREFIX ""
  CACHE STRING "Prefix path passed to Python's setup.py.")

IF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)
  SET(_EXTERNAL_LIBS)
  SET(_EXTERNAL_DIRS)

  IF(CUDA_FOUND AND WITH_CUDA)
    CUDA_ADD_LIBRARY(nnutils_ STATIC
      ../nnutils/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.cc

      ../nnutils/gpu/mask_image_from_size.h
      src/gpu/mask_image_from_size.h
      src/gpu/mask_image_from_size.cu

      ../nnutils/cpu/adaptive_avgpool_2d.h
      src/cpu/adaptive_avgpool_2d.h
      src/cpu/adaptive_avgpool_2d.cc

      ../nnutils/gpu/adaptive_avgpool_2d.h
      src/gpu/adaptive_avgpool_2d.h
      src/gpu/adaptive_avgpool_2d.cu)
    TARGET_INCLUDE_DIRECTORIES(nnutils_
      PRIVATE "${PYTORCH_INCLUDE_DIR}"
      PRIVATE "${PYTORCH_INCLUDE_DIR}/TH"
      PRIVATE "${PYTORCH_INCLUDE_DIR}/THC")
  ELSE()
    ADD_LIBRARY(nnutils_ STATIC
      ../nnutils/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.h
      src/cpu/mask_image_from_size.cc

      ../nnutils/cpu/adaptive_avgpool_2d.h
      src/cpu/adaptive_avgpool_2d.h
      src/cpu/adaptive_avgpool_2d.cc)
    TARGET_INCLUDE_DIRECTORIES(nnutils_
      PRIVATE "${PYTORCH_INCLUDE_DIR}"
      PRIVATE "${PYTORCH_INCLUDE_DIR}/TH")
    LIST(APPEND _EXTERNAL_LIBS ${PYTORCH_TH_LIBRARY})
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
    DEPENDS nnutils_pytorch/__init__.py
    DEPENDS nnutils_pytorch/nnutils.py
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libnnutils_.a
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  # Custom target to run setup.py build during make
  ADD_CUSTOM_TARGET(nnutils_pytorch ALL DEPENDS "${OUTPUT}" nnutils_)

  # This command will "install" nnutils_pytorch after it's built into a
  # temporal directory, in order to be able to run the Python tests without
  # actually installing the library.
  ADD_CUSTOM_COMMAND(TARGET nnutils_pytorch POST_BUILD
    COMMAND ${PYTHON_EXECUTABLE} "${SETUP_PY}"
    install --prefix=${CMAKE_CURRENT_BINARY_DIR}/inst
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)

  SET(_PYTHONPATH
    $ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}/build/lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages)

  ADD_TEST(NAME pytorch_nnutils_test
    COMMAND ${CMAKE_COMMAND} -E env "PYHONPATH=${_PYTHONPATH}"
    ${PYTHON_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/test/mask_image_from_size_test.py)

  # If no PYTORCH_SETUP_PREFIX variable is given by the user, we will use
  # CMAKE_INSTALL_PREFIX (if given).
  IF(NOT PYTORCH_SETUP_PREFIX)
    SET(PYTORCH_SETUP_PREFIX "${CMAKE_INSTALL_PREFIX}")
  ENDIF()
  MESSAGE(STATUS ${PYTORCH_SETUP_PREFIX})

  # make install will just execute python setup.py.
  # The install prefix will be PYTORCH_SETUP_PREFIX, if given. Or the system's
  # default if not given.
  IF(PYTORCH_SETUP_PREFIX)
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install --prefix=${PYTORCH_SETUP_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ELSE()
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ENDIF(PYTORCH_SETUP_PREFIX)
ENDIF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)

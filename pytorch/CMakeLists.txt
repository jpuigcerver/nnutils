SET(PYTORCH_SETUP_PREFIX ""
  CACHE STRING "Prefix path passed to Python's setup.py.")

ADD_SUBDIRECTORY(src)

IF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)
  SET(_EXTERNAL_LIBS)
  SET(_EXTERNAL_DIRS)

  IF(NOT PYTORCH_CUDA)
    MESSAGE(WARNING "Your PyTorch installation supports CUDA, but "
      "nnutils_pytorch won't. Are you sure about this?")
  ENDIF(NOT PYTORCH_CUDA)

  # Dependency with the wrapper code.
  LIST(APPEND _EXTERNAL_LIBS
    ${CMAKE_CURRENT_BINARY_DIR}/src/cpu/libpytorch_cpu_.a)
  IF(PYTORCH_CUDA)
    LIST(APPEND _EXTERNAL_LIBS
      ${CMAKE_CURRENT_BINARY_DIR}/src/gpu/libpytorch_gpu_.a)
  ENDIF(PYTORCH_CUDA)

  # Dependency with other libraries needed by the cpu and gpu implementations.
  GET_PROPERTY(_AUX_LIBS TARGET pytorch_cpu_
    PROPERTY LINK_LIBRARIES)
  GET_PROPERTY(_AUX_DIRS TARGET pytorch_cpu_
    PROPERTY INCLUDE_DIRECTORIES)
  LIST(APPEND _EXTERNAL_LIBS "${_AUX_LIBS}")
  LIST(APPEND _EXTERNAL_DIRS "${_AUX_DIRS}")
  IF(PYTORCH_CUDA)
    GET_PROPERTY(_AUX_LIBS TARGET pytorch_gpu_
      PROPERTY LINK_LIBRARIES)
    GET_PROPERTY(_AUX_DIRS TARGET pytorch_gpu_
      PROPERTY INCLUDE_DIRECTORIES)
    LIST(APPEND _EXTERNAL_LIBS "${_AUX_LIBS}")
    LIST(APPEND _EXTERNAL_DIRS "${_AUX_DIRS}")
  ENDIF(PYTORCH_CUDA)

  # A single target that compiles both pytorch_cpu_ and pytorch_gpu_
  ADD_CUSTOM_TARGET(pytorch_ ALL DEPENDS pytorch_cpu_)
  IF(PYTORCH_CUDA)
    ADD_DEPENDENCIES(pytorch_ pytorch_gpu_)
  ENDIF(PYTORCH_CUDA)

  SET(_INSTALL_DIR_TMP
    "${CMAKE_CURRENT_BINARY_DIR}/build/lib.linux-x86_64-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  SET(_PYTHONPATH "${_INSTALL_DIR_TMP}:$ENV{PYTHONPATH}")
  MESSAGE(STATUS "PyTorch tests PYTHONPATH=${_PYTHONPATH}")

  # Copy files to binary dir
  SET(_COPY_FILES)
  LIST(APPEND _COPY_FILES nnutils_pytorch/nnutils.py)
  LIST(APPEND _COPY_FILES nnutils_pytorch/mask_image_from_size_test.py)
  LIST(APPEND _COPY_FILES nnutils_pytorch/adaptive_maxpool_2d_test.py)
  LIST(APPEND _COPY_FILES nnutils_pytorch/adaptive_avgpool_2d_test.py)
  FOREACH(_file IN LISTS _COPY_FILES)
    CONFIGURE_FILE("${CMAKE_CURRENT_SOURCE_DIR}/${_file}"
      "${CMAKE_CURRENT_BINARY_DIR}/${_file}" COPYONLY)
  ENDFOREACH(_file IN LISTS _COPY_FILES)

  # Configure files
  SET(INIT_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/nnutils_pytorch/__init__.py.in")
  SET(INIT_PY "${CMAKE_CURRENT_BINARY_DIR}/nnutils_pytorch/__init__.py")
  SET(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  SET(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  SET(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

  IF(PYTORCH_CUDA)
    SET(NNUTILS_PYTORCH_PACKAGE
      "nnutils_pytorch_cu${CUDA_VERSION_MAJOR}${CUDA_VERSION_MINOR}")
  ELSE()
    SET(NNUTILS_PYTORCH_PACKAGE nnutils_pytorch)
  ENDIF(PYTORCH_CUDA)

  CONFIGURE_FILE("${INIT_PY_IN}" "${INIT_PY}")
  CONFIGURE_FILE("${SETUP_PY_IN}" "${SETUP_PY}")
  ADD_CUSTOM_COMMAND(OUTPUT "${OUTPUT}"
    COMMAND ${PYTHON_EXECUTABLE} setup.py build
    COMMAND ${CMAKE_COMMAND} -E touch "${OUTPUT}"
    DEPENDS "${INIT_PY}"
    DEPENDS "${NNUTILS_PY}"
    DEPENDS pytorch_
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  # Custom target to run setup.py build during make
  ADD_CUSTOM_TARGET(nnutils_pytorch ALL DEPENDS "${OUTPUT}")

  # Add tests
  ADD_TEST(NAME pytorch_mask_image_from_size_test
    COMMAND ${PYTHON_EXECUTABLE}
    nnutils_pytorch/mask_image_from_size_test.py)
  ADD_TEST(NAME pytorch_adaptive_avgpool_2d_test
    COMMAND ${PYTHON_EXECUTABLE}
    nnutils_pytorch/adaptive_avgpool_2d_test.py)
  ADD_TEST(NAME pytorch_adaptive_maxpool_2d_test
    COMMAND ${PYTHON_EXECUTABLE}
    nnutils_pytorch/adaptive_maxpool_2d_test.py)
  SET_TESTS_PROPERTIES(
    pytorch_mask_image_from_size_test
    pytorch_adaptive_avgpool_2d_test
    pytorch_adaptive_maxpool_2d_test
    PROPERTIES
    ENVIRONMENT PYTHONPATH=${_PYTHONPATH})
  # If no PYTORCH_SETUP_PREFIX variable is given by the user, we will use
  # CMAKE_INSTALL_PREFIX (if given).
  IF(NOT PYTORCH_SETUP_PREFIX)
    SET(PYTORCH_SETUP_PREFIX "${CMAKE_INSTALL_PREFIX}")
  ENDIF()

  # make install will just execute python setup.py.
  # The install prefix will be PYTORCH_SETUP_PREFIX, if given. Or the system's
  # default if not given.
  IF(PYTORCH_SETUP_PREFIX)
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} setup.py install --prefix=${PYTORCH_SETUP_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ELSE()
    INSTALL(
      CODE "EXECUTE_PROCESS(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
  ENDIF(PYTORCH_SETUP_PREFIX)

  SET_PROPERTY(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/build")
ENDIF(PYTORCH_FOUND AND WITH_PYTORCH AND PYTHONINTERP_FOUND)

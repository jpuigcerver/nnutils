IF(GTEST_FOUND AND WITH_TESTS)
  INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})
  INCLUDE_DIRECTORIES(/opt/torch_cuda-7.5/install/include/)
  INCLUDE_DIRECTORIES(/opt/torch_cuda-7.5/install/include/TH)
  INCLUDE_DIRECTORIES(/opt/torch_cuda-7.5/install/include/THC)

  ADD_EXECUTABLE(THTensorTest
    THTensorTest.cc
    THTensor.h THTraits.h generic/THTensor.h)
  TARGET_LINK_LIBRARIES(THTensorTest ${GTEST_BOTH_LIBRARIES})
  TARGET_LINK_LIBRARIES(THTensorTest /opt/torch_cuda-7.5/install/lib/libTH.so.0)
  ADD_TEST(THTensorTest THTensorTest)

  IF(CUDA_FOUND AND WITH_CUDA)
    CUDA_ADD_EXECUTABLE(THCTensorTest
      THCTensorTest.cc
      THTensor.h THTraits.h generic/THTensor.h
      THCTensor.h THCTraits.h generic/THCTensor.h)
    TARGET_LINK_LIBRARIES(THCTensorTest ${GTEST_LIBRARIES})
    TARGET_LINK_LIBRARIES(THCTensorTest
      /opt/torch_cuda-7.5/install/lib/libTH.so.0
      /opt/torch_cuda-7.5/install/lib/libTHC.so.0)
    ADD_TEST(THCTensorTest THCTensorTest)
  ENDIF(CUDA_FOUND AND WITH_CUDA)
ENDIF(GTEST_FOUND AND WITH_TESTS)

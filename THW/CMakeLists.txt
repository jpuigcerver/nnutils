# TODO(jpuigcerver): These tests will only work if PyTorch is
# installed. Allow to use the TH and THC in Torch as well.
IF(GTEST_FOUND AND WITH_TESTS)
  INCLUDE_DIRECTORIES(
    "${GTEST_INCLUDE_DIRS}"
    "${PYTORCH_INCLUDE_DIR}"
    "${PYTORCH_INCLUDE_DIR}/TH")

  ADD_EXECUTABLE(THTensorTest
    THTensorTest.cc
    THTensor.h THTraits.h generic/THTensor.h)
  TARGET_LINK_LIBRARIES(THTensorTest
    "${PYTORCH_TH_LIBRARY}"
    "${GTEST_BOTH_LIBRARIES}")
  ADD_TEST(THTensorTest THTensorTest)

  IF(CUDA_FOUND AND WITH_CUDA)
    INCLUDE_DIRECTORIES("${PYTORCH_INCLUDE_DIR}/THC")
    CUDA_ADD_EXECUTABLE(THCTensorTest
      THCTensorTest.cc
      THTensor.h THTraits.h generic/THTensor.h
      THCTensor.h THCTraits.h generic/THCTensor.h)
    TARGET_LINK_LIBRARIES(THCTensorTest
      "${PYTORCH_TH_LIBRARY}"
      "${PYTORCH_THC_LIBRARY}"
      "${GTEST_BOTH_LIBRARIES}")
    ADD_TEST(THCTensorTest THCTensorTest)
  ENDIF(CUDA_FOUND AND WITH_CUDA)
ENDIF(GTEST_FOUND AND WITH_TESTS)

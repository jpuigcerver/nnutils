#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <THW/THTensor.h>

using nnutils::THW::ConstTensor;
using nnutils::THW::MutableTensor;

namespace C {

template <typename THTensor>
THTensor* THTensor_newWithSize2d(int s1, int s2);

template <typename THTensor>
void THTensor_free(THTensor* tensor);

template <>
THByteTensor* THTensor_newWithSize2d<THByteTensor>(int s1, int s2) {
  return THByteTensor_newWithSize2d(s1, s2);
}

template <>
THCharTensor* THTensor_newWithSize2d<THCharTensor>(int s1, int s2) {
  return THCharTensor_newWithSize2d(s1, s2);
}

template <>
THShortTensor* THTensor_newWithSize2d<THShortTensor>(int s1, int s2) {
  return THShortTensor_newWithSize2d(s1, s2);
}

template <>
THIntTensor* THTensor_newWithSize2d<THIntTensor>(int s1, int s2) {
  return THIntTensor_newWithSize2d(s1, s2);
}

template <>
THLongTensor* THTensor_newWithSize2d<THLongTensor>(int s1, int s2) {
  return THLongTensor_newWithSize2d(s1, s2);
}

template <>
THFloatTensor* THTensor_newWithSize2d<THFloatTensor>(int s1, int s2) {
  return THFloatTensor_newWithSize2d(s1, s2);
}

template <>
THDoubleTensor* THTensor_newWithSize2d<THDoubleTensor>(int s1, int s2) {
  return THDoubleTensor_newWithSize2d(s1, s2);
}

template <>
void THTensor_free<THByteTensor>(THByteTensor* tensor) {
  THByteTensor_free(tensor);
}

template <>
void THTensor_free<THCharTensor>(THCharTensor* tensor) {
  THCharTensor_free(tensor);
}

template <>
void THTensor_free<THShortTensor>(THShortTensor* tensor) {
  THShortTensor_free(tensor);
}

template <>
void THTensor_free<THIntTensor>(THIntTensor* tensor) {
  THIntTensor_free(tensor);
}

template <>
void THTensor_free<THLongTensor>(THLongTensor* tensor) {
  THLongTensor_free(tensor);
}

template <>
void THTensor_free<THFloatTensor>(THFloatTensor* tensor) {
  THFloatTensor_free(tensor);
}

template <>
void THTensor_free<THDoubleTensor>(THDoubleTensor* tensor) {
  THDoubleTensor_free(tensor);
}

}  // namespace C

template <typename THTensor>
class TensorTest : public ::testing::Test {};

typedef ::testing::Types<THByteTensor,
                         THCharTensor,
                         THShortTensor,
                         THIntTensor,
                         THLongTensor,
                         THFloatTensor,
                         THDoubleTensor> TensorTypes;
TYPED_TEST_CASE(TensorTest, TensorTypes);

TYPED_TEST(TensorTest, Constructor) {
  TypeParam* tensor = C::THTensor_newWithSize2d<TypeParam>(5, 3);
  ConstTensor<TypeParam> ct(tensor);

  EXPECT_EQ(tensor->storage->data + tensor->storageOffset,
            ct.Data());

  EXPECT_EQ(2, ct.Dims());

  EXPECT_EQ(15, ct.Elems());

  EXPECT_EQ(5, ct.Size(0));
  EXPECT_EQ(3, ct.Size(1));

  EXPECT_EQ(3, ct.Stride(0));
  EXPECT_EQ(1, ct.Stride(1));

  C::THTensor_free(tensor);
}

TYPED_TEST(TensorTest, IsContiguous) {
  TypeParam* tensor = C::THTensor_newWithSize2d<TypeParam>(5, 3);
  MutableTensor<TypeParam> mt(tensor);

  EXPECT_TRUE(mt.IsContiguous());
  mt.Transpose(0, 1);
  EXPECT_FALSE(mt.IsContiguous());

  C::THTensor_free(tensor);
}

TYPED_TEST(TensorTest, IsSameSizeAs) {
  TypeParam* tensorA = C::THTensor_newWithSize2d<TypeParam>(5, 3);
  TypeParam* tensorB = C::THTensor_newWithSize2d<TypeParam>(1, 3);
  THByteTensor*  tensorC = C::THTensor_newWithSize2d<THByteTensor>(5, 3);
  THFloatTensor* tensorD = C::THTensor_newWithSize2d<THFloatTensor>(5, 1);
  ConstTensor<TypeParam> ctA(tensorA);
  ConstTensor<TypeParam> ctB(tensorB);
  ConstTensor<THByteTensor> ctC(tensorC);
  ConstTensor<THFloatTensor> ctD(tensorD);

  EXPECT_TRUE(ctA.IsSameSizeAs(ctA));   // same size, same type
  EXPECT_TRUE(ctA.IsSameSizeAs(ctC));   // same size, diff type

  EXPECT_FALSE(ctA.IsSameSizeAs(ctB));  // diff size, same type
  EXPECT_FALSE(ctA.IsSameSizeAs(ctD));  // diff size, diff type

  C::THTensor_free(tensorA);
  C::THTensor_free(tensorB);
  C::THTensor_free(tensorC);
  C::THTensor_free(tensorD);
}

TYPED_TEST(TensorTest, Fill) {
  typedef typename MutableTensor<TypeParam>::DType DType;

  TypeParam* tensor = C::THTensor_newWithSize2d<TypeParam>(2, 3);
  MutableTensor<TypeParam> mt(tensor);
  mt.Fill(3);

  EXPECT_THAT(std::vector<DType>(mt.Data(), mt.Data() + 6), ::testing::Each(3));

  C::THTensor_free(tensor);
}

TYPED_TEST(TensorTest, Resize) {
  TypeParam* tensor = C::THTensor_newWithSize2d<TypeParam>(2, 3);
  MutableTensor<TypeParam> mt(tensor);

  mt.Resize({4, 3, 2, 1});
  EXPECT_EQ(4, mt.Dims());
  EXPECT_EQ(4, mt.Size(0));
  EXPECT_EQ(3, mt.Size(1));
  EXPECT_EQ(2, mt.Size(2));
  EXPECT_EQ(1, mt.Size(3));

  C::THTensor_free(tensor);
}

TYPED_TEST(TensorTest, ResizeAs) {
  TypeParam* tensorA = C::THTensor_newWithSize2d<TypeParam>(5, 3);
  THFloatTensor* tensorB = C::THTensor_newWithSize2d<THFloatTensor>(1, 1);

  MutableTensor<TypeParam> mtA(tensorA);
  MutableTensor<THFloatTensor> mtB(tensorB);

  mtB.ResizeAs(mtA);
  EXPECT_TRUE(mtB.IsSameSizeAs(mtA));
  EXPECT_TRUE(mtA.IsSameSizeAs(mtB));

  C::THTensor_free(tensorA);
  C::THTensor_free(tensorB);
}

TYPED_TEST(TensorTest, Zero) {
  typedef typename MutableTensor<TypeParam>::DType DType;

  TypeParam* tensor = C::THTensor_newWithSize2d<TypeParam>(2, 3);
  MutableTensor<TypeParam> mt(tensor);
  mt.Fill(3);
  mt.Zero();
  EXPECT_THAT(std::vector<DType>(mt.Data(), mt.Data() + 6), ::testing::Each(0));

  C::THTensor_free(tensor);
}

import numpy as np
import torch
import unittest

from torch.autograd import Variable
from nnutils_pytorch import mask_image_from_size

class MaskImageFromSizeTest(unittest.TestCase):
    def setUp(self):
        self._x = Variable(torch.Tensor([[1, 2, 3 ,4],
                                         [5, 6, 7, 8]]).resize_(2, 1, 1, 4),
                           requires_grad=True)
        self._s = Variable(torch.LongTensor([[1, 2],
                                             [1, 3]]))
        self._dy = Variable(torch.Tensor([[8, 7, 6, 5],
                                          [4, 3, 2, 1]]).resize_(2, 1, 1, 4))

        self._expect_y = torch.Tensor([[1, 2, -1, -1],
                                       [5, 6, 7, -1]]).resize_(2, 1, 1, 4)
        self._expect_dx = torch.Tensor([[8, 7, 0, 0],
                                        [4, 3, 2, 0]]).resize_(2, 1, 1, 4)

    def convert(self, cuda, dtype):
        self._x = self._x.type(dtype)
        self._dy = self._dy.type(dtype)
        self._expect_y = self._expect_y.type(dtype)
        self._expect_dx = self._expect_dx.type(dtype)

        if cuda:
            self._x = self._x.cuda()
            self._s = self._s.cuda()
            self._dy = self._dy.cuda()
        else:
            self._x = self._x.cpu()
            self._s = self._s.cpu()
            self._dy = self._dy.cpu()

    def run_base_test(self):
        f = mask_image_from_size(-1)
        y = f(self._x, self._s)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y)

        dx, _ = f.backward(self._dy)
        np.testing.assert_array_almost_equal(dx.data.cpu(), self._expect_dx)

    def test_cpu_f32(self):
        self.convert(False, 'torch.FloatTensor')
        self.run_base_test()

    def test_cpu_f64(self):
        self.convert(False, 'torch.DoubleTensor')
        self.run_base_test()

    def test_cpu_s16(self):
        self.convert(False, 'torch.ShortTensor')
        self.run_base_test()

    def test_cpu_s32(self):
        self.convert(False, 'torch.IntTensor')
        self.run_base_test()

    def test_cpu_s64(self):
        self.convert(False, 'torch.LongTensor')
        self.run_base_test()

def test_gpu_f32(self):
    self.convert(True, 'torch.FloatTensor')
    self.run_base_test()

def test_gpu_f64(self):
    self.convert(True, 'torch.DoubleTensor')
    self.run_base_test()

def test_gpu_s16(self):
    self.convert(True, 'torch.ShortTensor')
    self.run_base_test()

def test_gpu_s32(self):
    self.convert(True, 'torch.IntTensor')
    self.run_base_test()

def test_gpu_s64(self):
    self.convert(True, 'torch.LongTensor')
    self.run_base_test()

# If cuda support is available, register tests to the class.
if torch.cuda.is_available():
    MaskImageFromSizeTest.test_gpu_f32 = test_gpu_f32
    MaskImageFromSizeTest.test_gpu_f64 = test_gpu_f64
    MaskImageFromSizeTest.test_gpu_s16 = test_gpu_s16
    MaskImageFromSizeTest.test_gpu_s32 = test_gpu_s32
    MaskImageFromSizeTest.test_gpu_s64 = test_gpu_s64

if __name__ == '__main__':
    unittest.main()

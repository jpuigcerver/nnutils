import numpy as np
import torch
import unittest

from torch.autograd import Variable
from nnutils_pytorch import adaptive_maxpool_2d

class AdaptiveMaxpool2dTest(unittest.TestCase):
    def setUp(self):
        self._s = torch.LongTensor([[3, 4],
                                    [2, 8]])

        self._x = torch.Tensor([
            # Img 1 (3 x 4)
            [[ 1,  2,  3,  4, 99, 99, 99, 99],
             [ 5,  6,  7,  8, 99, 99, 99, 99],
             [ 9, 10, 11, 12, 99, 99, 99, 99]],
            # Img 2 (2 x 8)
            [[ 1,  2,  3,  4,  5,  6,  7,  8],
             [ 9, 10, 11, 12, 13, 14, 15, 16],
             [99, 99, 99, 99, 99, 99, 99, 99]]
            ]).resize_(2, 1, 3, 8)

        self._dy = torch.Tensor([
            # Output gradient w.r.t Image 1
            [[3, 6, 9, 12]],
            # Output gradient w.r.t. Image 2
            [[8, 12, 16, 20]]
        ]).resize_(2, 1, 1, 4)

        self._expect_y = torch.Tensor([
            # Expected output 1
            [[9, 10, 11, 12]],
            # Expected output 2
            [[10, 12, 14, 16]]
        ]).resize_(2, 1, 1, 4)

        self._expect_dx = torch.Tensor([
            # Input gradient w.r.t. Image 1
            [[  0,  0,  0,  0, 0, 0, 0, 0],
             [  0,  0,  0,  0, 0, 0, 0, 0],
             [  3,  6,  9, 12, 0, 0, 0, 0]],
            # Input gradient w.r.t. Image 2
            [[0, 0, 0,  0, 0,  0, 0,  0],
             [0, 8, 0, 12, 0, 16, 0, 20],
             [0, 0, 0,  0, 0,  0, 0, 0]]
        ]).resize_(2, 1, 3, 8)


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
        x = Variable(self._x, requires_grad=True)
        xs = Variable(self._s, requires_grad=False)
        y = adaptive_maxpool_2d(x, output_sizes=(1, 4), batch_sizes=xs)
        y.backward(self._dy, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y)
        np.testing.assert_array_almost_equal(x.grad.data.cpu(), self._expect_dx)

    def test_cpu_f32(self):
        self.convert(False, 'torch.FloatTensor')
        self.run_base_test()

    def test_cpu_f64(self):
        self.convert(False, 'torch.DoubleTensor')
        self.run_base_test()

def test_gpu_f32(self):
    self.convert(True, 'torch.FloatTensor')
    self.run_base_test()

def test_gpu_f64(self):
    self.convert(True, 'torch.DoubleTensor')
    self.run_base_test()

# If cuda support is available, register tests to the class.
if torch.cuda.is_available():
    AdaptiveMaxpool2dTest.test_gpu_f32 = test_gpu_f32
    AdaptiveMaxpool2dTest.test_gpu_f64 = test_gpu_f64

if __name__ == '__main__':
    unittest.main()

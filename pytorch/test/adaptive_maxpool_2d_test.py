import numpy as np
import torch
import unittest

from torch.autograd import Variable
from nnutils_pytorch import is_cuda_available, adaptive_maxpool_2d

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

        self._dy_fixed_height = torch.Tensor([
            # Output gradient w.r.t. Image 1
            [[2, 4, 6, 8, 0,  0,  0,  0]],
            # Output gradient w.r.t. Image 2
            [[1, 3, 5, 7, 9, 11, 13, 15]],
        ]).resize_(2, 1, 1, 8)

        self._dy_fixed_width = torch.Tensor([
            # Output gradient w.r.t. Image 1
            [[ 2,  4],
             [ 6,  8],
             [10, 12]],
            # Output gradient w.r.t. Image 2
            [[1, 3],
             [5, 7],
             [0, 0]],
        ]).resize_(2, 1, 3, 2)

        self._expect_y = torch.Tensor([
            # Expected output 1
            [[9, 10, 11, 12]],
            # Expected output 2
            [[10, 12, 14, 16]]
        ]).resize_(2, 1, 1, 4)

        self._expect_y_fixed_height = torch.Tensor([
            # Expected output 1
            [[9, 10, 11, 12, 0, 0, 0, 0]],
            # Expected output 2
            [[9, 10, 11, 12, 13, 14, 15, 16]],
        ]).resize_(2, 1, 1, 8)

        self._expect_y_fixed_width = torch.Tensor([
            # Expected output 1
            [[ 2, 4],
             [ 6,  8],
             [10, 12]],
            # Expected output 2
            [[ 4,  8],
             [12, 16],
             [ 0,  0]],
        ]).resize_(2, 1, 3, 2)

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

        self._expect_dx_fixed_height = torch.Tensor([
            # Input gradient w.r.t. Image 1
            [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [2, 4, 6, 8, 0, 0, 0, 0]],
            # Input gradient w.r.t. Image 2
            [[0, 0, 0, 0, 0,  0,  0,  0],
             [1, 3, 5, 7, 9, 11, 13, 15],
             [0, 0, 0, 0, 0,  0,  0,  0]]
        ]).resize_(2, 1, 3, 8)

        self._expect_dx_fixed_width = torch.Tensor([
            # Input gradient w.r.t. Image 1
            [[0,  2, 0,  4, 0, 0, 0, 0],
             [0,  6, 0,  8, 0, 0, 0, 0],
             [0, 10, 0, 12, 0, 0, 0, 0]],
            # Input gradient w.r.t. Image 2
            [[0, 0, 0, 1, 0, 0, 0, 3],
             [0, 0, 0, 5, 0, 0, 0, 7],
             [0, 0, 0, 0, 0, 0, 0, 0]]
        ]).resize_(2, 1, 3, 8)

    def convert(self, cuda, dtype):
        self._x = self._x.type(dtype)
        self._dy = self._dy.type(dtype)
        self._dy_fixed_height = self._dy_fixed_height.type(dtype)
        self._dy_fixed_width = self._dy_fixed_width.type(dtype)
        self._expect_y = self._expect_y.type(dtype)
        self._expect_y_fixed_height = self._expect_y_fixed_height.type(dtype)
        self._expect_y_fixed_width = self._expect_y_fixed_width.type(dtype)
        self._expect_dx = self._expect_dx.type(dtype)
        self._expect_dx_fixed_height = self._expect_dx_fixed_height.type(dtype)
        self._expect_dx_fixed_width = self._expect_dx_fixed_width.type(dtype)
        if cuda:
            self._x = self._x.cuda()
            self._s = self._s.cuda()
            self._dy = self._dy.cuda()
            self._dy_fixed_height = self._dy_fixed_height.cuda()
            self._dy_fixed_width = self._dy_fixed_width.cuda()
        else:
            self._x = self._x.cpu()
            self._s = self._s.cpu()
            self._dy = self._dy.cpu()
            self._dy_fixed_height = self._dy_fixed_height.cpu()
            self._dy_fixed_width = self._dy_fixed_width.cpu()

    def run_base(self, cuda, ttype):
        self.convert(cuda, ttype)
        x = Variable(self._x, requires_grad=True)
        xs = Variable(self._s, requires_grad=False)
        y = adaptive_maxpool_2d(x, output_sizes=(1, 4), batch_sizes=xs)
        y.backward(self._dy, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y)
        np.testing.assert_array_almost_equal(x.grad.data.cpu(), self._expect_dx)

    def run_fixed_height(self, cuda, ttype):
        self.convert(cuda, ttype)
        x = Variable(self._x, requires_grad=True)
        xs = Variable(self._s, requires_grad=False)
        y = adaptive_maxpool_2d(x, output_sizes=(1, None), batch_sizes=xs)
        y.backward(self._dy_fixed_height, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y_fixed_height)
        np.testing.assert_array_almost_equal(x.grad.data.cpu(), self._expect_dx_fixed_height)

    def run_fixed_width(self, cuda, ttype):
        self.convert(cuda, ttype)
        x = Variable(self._x, requires_grad=True)
        xs = Variable(self._s, requires_grad=False)
        y = adaptive_maxpool_2d(x, output_sizes=(None, 2), batch_sizes=xs)
        y.backward(self._dy_fixed_width, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y_fixed_width)
        np.testing.assert_array_almost_equal(x.grad.data.cpu(), self._expect_dx_fixed_width)


# Register tests for different types, and different devices.
for ttype, dtype in zip(['torch.FloatTensor', 'torch.DoubleTensor'],
                        ['f32', 'f64']):
    setattr(AdaptiveMaxpool2dTest,
            'test_cpu_%s' % dtype,
            lambda self: self.run_base(False, ttype))
    setattr(AdaptiveMaxpool2dTest,
            'test_fixed_height_cpu_%s' % dtype,
            lambda self: self.run_fixed_height(False, ttype))
    setattr(AdaptiveMaxpool2dTest,
            'test_fixed_width_cpu_%s' % dtype,
            lambda self: self.run_fixed_width(False, ttype))
    if is_cuda_available():
        setattr(AdaptiveMaxpool2dTest,
                'test_gpu_%s' % dtype,
                lambda self: self.run_base(True, ttype))
        setattr(AdaptiveMaxpool2dTest,
                'test_fixed_height_gpu_%s' % dtype,
                lambda self: self.run_fixed_height(True, ttype))
        setattr(AdaptiveMaxpool2dTest,
                'test_fixed_width_gpu_%s' % dtype,
                lambda self: self.run_fixed_width(True, ttype))

if __name__ == '__main__':
    unittest.main()

from __future__ import absolute_import

import numpy as np
import torch
import unittest

from nnutils_pytorch import adaptive_avgpool_2d
from torch.nn.functional import adaptive_avg_pool2d as torch_adaptive_avg_pool2d


class AdaptiveAvgpool2dTest(unittest.TestCase):
    def setUp(self):
        self._s = torch.LongTensor([[3, 4], [2, 8]])

        self._x = torch.Tensor(
            [
                # Img 1 (3 x 4)
                [
                    [1, 2, 3, 4, 99, 99, 99, 99],
                    [5, 6, 7, 8, 99, 99, 99, 99],
                    [9, 10, 11, 12, 99, 99, 99, 99],
                ],
                # Img 2 (2 x 8)
                [
                    [1, 2, 3, 4, 5, 6, 7, 8],
                    [9, 10, 11, 12, 13, 14, 15, 16],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                ],
            ]
        ).resize_(2, 1, 3, 8)

        self._dy = torch.Tensor(
            [
                # Output gradient w.r.t Image 1
                [[3, 6, 9, 12]],
                # Output gradient w.r.t. Image 2
                [[8, 12, 16, 20]],
            ]
        ).resize_(2, 1, 1, 4)

        self._dy_fixed_height = torch.Tensor(
            [
                # Output gradient w.r.t. Image 1
                [[3, 6, 9, 12, 0, 0, 0, 0]],
                # Output gradient w.r.t. Image 2
                [[6, 8, 10, 12, 14, 16, 18, 20]],
            ]
        ).resize_(2, 1, 1, 8)

        self._dy_fixed_width = torch.Tensor(
            [
                # Output gradient w.r.t. Image 1
                [[2, 4], [6, 8], [10, 12]],
                # Output gradient w.r.t. Image 2
                [[4, 8], [12, 16], [0, 0]],
            ]
        ).resize_(2, 1, 3, 2)

        self._expect_y = torch.Tensor(
            [
                # Expected output 1
                [[5, 6, 7, 8]],
                # Expected output 2
                [[5.5, 7.5, 9.5, 11.5]],
            ]
        ).resize_(2, 1, 1, 4)

        self._expect_y_fixed_height = torch.Tensor(
            [
                # Expected output 1
                [[5, 6, 7, 8, 0, 0, 0, 0]],
                # Expected output 2
                [[5, 6, 7, 8, 9, 10, 11, 12]],
            ]
        ).resize_(2, 1, 1, 8)

        self._expect_y_fixed_width = torch.Tensor(
            [
                # Expected output 1
                [[1.5, 3.5], [5.5, 7.5], [9.5, 11.5]],
                # Expected output 2
                [[2.5, 6.5], [10.5, 14.5], [0, 0]],
            ]
        ).resize_(2, 1, 3, 2)

        self._expect_dx = torch.Tensor(
            [
                # Input gradient w.r.t. Image 1
                [
                    [1, 2, 3, 4, 0, 0, 0, 0],
                    [1, 2, 3, 4, 0, 0, 0, 0],
                    [1, 2, 3, 4, 0, 0, 0, 0],
                ],
                # Input gradient w.r.t. Image 2
                [
                    [2, 2, 3, 3, 4, 4, 5, 5],
                    [2, 2, 3, 3, 4, 4, 5, 5],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        ).resize_(2, 1, 3, 8)

        self._expect_dx_fixed_height = torch.Tensor(
            [
                # Input gradient w.r.t. Image 1
                [
                    [1, 2, 3, 4, 0, 0, 0, 0],
                    [1, 2, 3, 4, 0, 0, 0, 0],
                    [1, 2, 3, 4, 0, 0, 0, 0],
                ],
                # Input gradient w.r.t. Image 2
                [
                    [3, 4, 5, 6, 7, 8, 9, 10],
                    [3, 4, 5, 6, 7, 8, 9, 10],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        ).resize_(2, 1, 3, 8)

        self._expect_dx_fixed_width = torch.Tensor(
            [
                # Input gradient w.r.t. Image 1
                [
                    [1, 1, 2, 2, 0, 0, 0, 0],
                    [3, 3, 4, 4, 0, 0, 0, 0],
                    [5, 5, 6, 6, 0, 0, 0, 0],
                ],
                # Input gradient w.r.t. Image 2
                [
                    [1, 1, 1, 1, 2, 2, 2, 2],
                    [3, 3, 3, 3, 4, 4, 4, 4],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
            ]
        ).resize_(2, 1, 3, 8)

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
        x = self._x.detach().requires_grad_()
        xs = self._s.detach()
        y = adaptive_avgpool_2d(x, output_sizes=(1, 4), batch_sizes=xs)
        y.backward(self._dy, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y)
        np.testing.assert_array_almost_equal(x.grad.data.cpu(), self._expect_dx)

    def run_fixed_height(self, cuda, ttype):
        self.convert(cuda, ttype)
        x = self._x.detach().requires_grad_()
        xs = self._s.detach()
        y = adaptive_avgpool_2d(x, output_sizes=(1, None), batch_sizes=xs)
        y.backward(self._dy_fixed_height, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y_fixed_height)
        np.testing.assert_array_almost_equal(
            x.grad.data.cpu(), self._expect_dx_fixed_height
        )

    def run_fixed_width(self, cuda, ttype):
        self.convert(cuda, ttype)
        x = self._x.detach().requires_grad_()
        xs = self._s.detach()
        y = adaptive_avgpool_2d(x, output_sizes=(None, 2), batch_sizes=xs)
        y.backward(self._dy_fixed_width, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y_fixed_width)
        np.testing.assert_array_almost_equal(
            x.grad.data.cpu(), self._expect_dx_fixed_width
        )

    @staticmethod
    def run_compare_reference_smaller_output(cuda, ttype):
        x3 = torch.randn(2, 3, 10, 15).type(ttype)
        xs3 = torch.LongTensor([[4, 5], [8, 6]])
        x1 = x3[0, :, :4, :5].clone().view(1, 3, 4, 5)
        x2 = x3[1, :, :8, :6].clone().view(1, 3, 8, 6)
        if cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
            xs3 = xs3.cuda()
        else:
            x1 = x1.cpu()
            x2 = x2.cpu()
            x3 = x3.cpu()
            xs3 = xs3.cpu()
        x1 = x1.requires_grad_()
        x2 = x2.requires_grad_()
        x3 = x3.requires_grad_()
        # Compare forward
        y1 = torch_adaptive_avg_pool2d(x1, output_size=(2, 3))
        y2 = torch_adaptive_avg_pool2d(x2, output_size=(2, 3))
        y3 = adaptive_avgpool_2d(x3, output_sizes=(2, 3), batch_sizes=xs3)
        np.testing.assert_almost_equal(
            y3.data.cpu().numpy(), torch.cat([y1, y2]).data.cpu().numpy()
        )
        # Compare backward
        dx1, dx2, = torch.autograd.grad(y1.sum() + y2.sum(), [x1, x2])
        dx3, = torch.autograd.grad(y3.sum(), [x3])
        ref = dx3.clone().zero_()
        ref[0, :, :4, :5] = dx1.data
        ref[1, :, :8, :6] = dx2.data
        np.testing.assert_almost_equal(dx3.data.cpu().numpy(), ref.data.cpu().numpy())

    @staticmethod
    def run_compare_reference_larger_output(cuda, ttype):
        x3 = torch.randn(2, 3, 10, 15).type(ttype)
        xs3 = torch.LongTensor([[4, 5], [8, 6]])
        x1 = x3[0, :, :4, :5].clone().view(1, 3, 4, 5)
        x2 = x3[1, :, :8, :6].clone().view(1, 3, 8, 6)
        if cuda:
            x1 = x1.cuda()
            x2 = x2.cuda()
            x3 = x3.cuda()
            xs3 = xs3.cuda()
        else:
            x1 = x1.cpu()
            x2 = x2.cpu()
            x3 = x3.cpu()
            xs3 = xs3.cpu()
        x1 = x1.requires_grad_()
        x2 = x2.requires_grad_()
        x3 = x3.requires_grad_()
        # Compare forward
        y1 = torch_adaptive_avg_pool2d(x1, output_size=(20, 25))
        y2 = torch_adaptive_avg_pool2d(x2, output_size=(20, 25))
        y3 = adaptive_avgpool_2d(x3, output_sizes=(20, 25), batch_sizes=xs3)
        np.testing.assert_almost_equal(
            y3.data.cpu().numpy(), torch.cat([y1, y2]).data.cpu().numpy()
        )
        # Compare backward
        dx1, dx2, = torch.autograd.grad(y1.sum() + y2.sum(), [x1, x2])
        dx3, = torch.autograd.grad(y3.sum(), [x3])
        ref = dx3.clone().zero_()
        ref[0, :, :4, :5] = dx1.data
        ref[1, :, :8, :6] = dx2.data
        np.testing.assert_almost_equal(dx3.data.cpu().numpy(), ref.data.cpu().numpy())


# Register tests for different types, and different devices.
types = [("torch.FloatTensor", "f32"), ("torch.DoubleTensor", "f64")]
devices = [("cpu", False)]
if torch.cuda.is_available():
    devices += [("gpu", True)]

for ttype, dtype in types:
    for device, use_cuda in devices:
        setattr(
            AdaptiveAvgpool2dTest,
            "test_%s_%s" % (device, dtype),
            lambda self: self.run_base(use_cuda, ttype),
        )
        setattr(
            AdaptiveAvgpool2dTest,
            "test_fixed_height_%s_%s" % (device, dtype),
            lambda self: self.run_fixed_height(use_cuda, ttype),
        )
        setattr(
            AdaptiveAvgpool2dTest,
            "test_fixed_width_%s_%s" % (device, dtype),
            lambda self: self.run_fixed_width(use_cuda, ttype),
        )
        setattr(
            AdaptiveAvgpool2dTest,
            "test_compare_to_reference_smaller_output_%s_%s" % (device, dtype),
            lambda self: self.run_compare_reference_smaller_output(use_cuda, ttype),
        )
        setattr(
            AdaptiveAvgpool2dTest,
            "test_compare_to_reference_larger_output_%s_%s" % (device, dtype),
            lambda self: self.run_compare_reference_larger_output(use_cuda, ttype),
        )

if __name__ == "__main__":
    unittest.main()

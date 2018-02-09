import numpy as np
import torch
import unittest

from torch.autograd import Variable
from nnutils_pytorch import mask_image_from_size

class MaskImageFromSizeTest(unittest.TestCase):
    def setUp(self):
        self._s = torch.LongTensor([[1, 2],
                                    [1, 3]])
        self._x = torch.Tensor([[1, 2, 3 ,4],
                                [5, 6, 7, 8]]).resize_(2, 1, 1, 4)

        self._dy = torch.Tensor([[8, 7, 6, 5],
                                 [4, 3, 2, 1]]).resize_(2, 1, 1, 4)

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

    def run_base(self, cuda, ttype):
        self.convert(cuda, ttype)
        x = Variable(self._x, requires_grad=True)
        xs = Variable(self._s, requires_grad=False)
        y = mask_image_from_size(x, xs, mask_value=-1)
        y.backward(self._dy, retain_graph=True)
        np.testing.assert_array_almost_equal(y.data.cpu(), self._expect_y)
        np.testing.assert_array_almost_equal(x.grad.data.cpu(), self._expect_dx)

# Register tests for different types, and different devices.
for ttype, dtype in zip(['torch.ShortTensor',
                         'torch.IntTensor',
                         'torch.LongTensor',
                         'torch.FloatTensor',
                         'torch.DoubleTensor'],
                        ['s16', 's32', 's64', 'f32', 'f64']):
    setattr(MaskImageFromSizeTest,
            'test_cpu_%s' % dtype,
            lambda self: self.run_base(False, ttype))
    if torch.cuda.is_available():
        setattr(MaskImageFromSizeTest,
                'test_gpu_%s' % dtype,
                lambda self: self.run_base(True, ttype))

if __name__ == '__main__':
    unittest.main()

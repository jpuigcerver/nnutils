from __future__ import absolute_import

import unittest

import torch
from nnutils_pytorch import mask_image_from_size
from torch.autograd import Variable


class MaskImageFromSizeTest(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(5, 3, 213, 217)
        self.xs = torch.LongTensor(
            [[29, 29], [213, 217], [213, 15], [15, 217], [97, 141]]
        )
        # Cost for each output pixel
        self.cy = torch.randn(5, 3, 213, 217)

    def convert(self, cuda, dtype):
        self.x = self.x.type(dtype)
        self.cy = self.cy.type(dtype)
        if cuda:
            self.x = self.x.cuda()
            self.xs = self.xs.cuda()
            self.cy = self.cy.cuda()
        else:
            self.x = self.x.cpu()
            self.xs = self.xs.cpu()
            self.cy = self.cy.cpu()

    def run_base(self, cuda, ttype):
        self.convert(cuda, ttype)
        x = self.x.detach().requires_grad_()
        cy = self.cy.detach()
        y = mask_image_from_size(batch_input=x, batch_sizes=self.xs, mask_value=99)
        # Check forward
        for i, (xi, yi, s) in enumerate(zip(self.x, y.data, self.xs)):
            # Check non-masked area
            d = torch.sum(yi[:, : s[0], : s[1]] != xi[:, : s[0], : s[1]])
            self.assertEqual(
                0, d, msg="Sample {} failed in the non-masked area".format(i)
            )
            # Check masked area
            d1 = torch.sum(yi[:, s[0] :, :] != 99) if s[0] < self.x.size(2) else 0
            d2 = torch.sum(yi[:, :, s[1] :] != 99) if s[1] < self.x.size(3) else 0
            self.assertEqual(
                0, d1 + d2, msg="Sample {} failed in the masked area".format(i)
            )
        # Check gradients
        cost = torch.sum(torch.mul(y, cy))
        dx, = torch.autograd.grad(cost, (x,))
        for i, (xi, yi, s) in enumerate(zip(dx.data, self.cy, self.xs)):
            # Check non-masked area
            d = torch.sum(xi[:, : s[0], : s[1]] != yi[:, : s[0], : s[1]])
            self.assertEqual(
                0, d, msg="Sample {} failed in the non-masked area".format(i)
            )
            # Check masked area
            d1 = torch.sum(xi[:, s[0] :, :] != 0) if s[0] < self.x.size(2) else 0
            d2 = torch.sum(xi[:, :, s[1] :] != 0) if s[1] < self.x.size(3) else 0
            self.assertEqual(
                0, d1 + d2, msg="Sample {} failed in the masked area".format(i)
            )


# Register tests for different types, and different devices.
for ttype, dtype in [
    ("torch.ByteTensor", "u8"),
    ("torch.ShortTensor", "s16"),
    ("torch.IntTensor", "s32"),
    ("torch.LongTensor", "s64"),
    ("torch.FloatTensor", "f32"),
    ("torch.DoubleTensor", "f64"),
]:
    setattr(
        MaskImageFromSizeTest,
        "test_cpu_%s" % dtype,
        lambda self: self.run_base(False, ttype),
    )
    if torch.cuda.is_available():
        setattr(
            MaskImageFromSizeTest,
            "test_gpu_%s" % dtype,
            lambda self: self.run_base(True, ttype),
        )

if __name__ == "__main__":
    unittest.main()

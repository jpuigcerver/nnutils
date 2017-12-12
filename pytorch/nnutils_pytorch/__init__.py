import torch
from torch.autograd import Function
from ._nnutils import lib as _lib, ffi as _ffi

class image_mask_from_size(Function):
    def forward(self, batch, batch_size, mask=0):
        is_cuda = True if batch.is_cuda else False
        print(batch.type())
        return batch

    def backward(self, grad_output):
        masked_grad = grad_output.clone()
        # TODO(joapuipe): Mask gradient
        return masked_grad, None, None

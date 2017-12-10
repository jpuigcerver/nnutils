import torch
from torch.autograd import Function

class ImageMaskFromSize(Function):
    def forward(self, batch, batch_size, mask=0):
        pass

    def backward(self, grad_output):
        masked_grad = grad_output.clone()
        # TODO(joapuipe): Mask gradient
        return masked_grad, None, None
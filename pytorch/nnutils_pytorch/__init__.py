import torch
from torch.autograd import Function, Variable
from torch.utils.ffi import _wrap_function
from ._nnutils import lib as _lib, ffi as _ffi

__all__ = []

def _import_symbols(loc):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        loc[symbol] = _wrap_function(fn, _ffi)
        __all__.append(symbol)
_import_symbols(locals())

_funcs = {
    'torch.FloatTensor'       : nnutils_mask_image_from_size_THFloatTensor,
    'torch.DoubleTensor'      : nnutils_mask_image_from_size_THDoubleTensor,
    'torch.cuda.FloatTensor'  : nnutils_mask_image_from_size_THCudaTensor,
    'torch.cuda.DoubleTensor' : nnutils_mask_image_from_size_THCudaDoubleTensor
}

class image_mask_from_size(Function):
    def __init__(self, mask_value=0, inplace=False):
        super(image_mask_from_size, self).__init__()
        self._mask_value = mask_value
        self._inplace = inplace

    def forward(self, batch, batch_size=None):
        self.save_for_backward(batch_size)
        if batch_size is None:
            return batch
        else:
            assert(batch.is_cuda == batch_size.is_cuda)
            masked_batch = batch if self._inplace else batch.clone()
            x = (masked_batch.data if isinstance(masked_batch, Variable)
                 else masked_batch)
            xs = (batch_size.data if isinstance(batch_size, Variable)
                  else batch_size)
            _funcs[masked_batch.type()](x, xs, self._mask_value)
            return masked_batch

    def backward(self, grad_output):
        batch_size, = self.saved_variables
        if batch_size is None:
            return grad_output
        else:
            assert(grad_output.is_cuda == batch_size.is_cuda)
            masked_grad = grad_output if self._inplace else grad_output.clone()
            x = (masked_grad.data if isinstance(masked_grad, Variable)
                 else masked_grad)
            xs = (batch_size.data if isinstance(batch_size, Variable)
                  else batch_size)
            _funcs[masked_grad.type()](x, xs, 0)
            return masked_grad, None

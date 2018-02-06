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

# If GPU implementations are not defined in the imgdistort.so, set the symbols
# to None.
for t in ['f32', 'f64', 'u8', 's8', 's16', 's32', 's64']:
    if ('nnutils_mask_image_from_size_gpu_%s' % t) not in locals():
        locals()['nnutils_mask_image_from_size_gpu_%s' % t] = None

_mask_image = {
    'torch.ByteTensor'   : nnutils_mask_image_from_size_cpu_u8,
    'torch.CharTensor'   : nnutils_mask_image_from_size_cpu_s8,
    'torch.ShortTensor'  : nnutils_mask_image_from_size_cpu_s16,
    'torch.IntTensor'    : nnutils_mask_image_from_size_cpu_s32,
    'torch.LongTensor'   : nnutils_mask_image_from_size_cpu_s64,
    'torch.FloatTensor'  : nnutils_mask_image_from_size_cpu_f32,
    'torch.DoubleTensor' : nnutils_mask_image_from_size_cpu_f64
}

if torch.cuda.is_available():
    _mask_image['torch.cuda.ByteTensor']   = (
        nnutils_mask_image_from_size_gpu_u8)
    _mask_image['torch.cuda.CharTensor']   = (
        nnutils_mask_image_from_size_gpu_s8)
    _mask_image['torch.cuda.ShortTensor']  = (
        nnutils_mask_image_from_size_gpu_s16)
    _mask_image['torch.cuda.IntTensor']    = (
        nnutils_mask_image_from_size_gpu_s32)
    _mask_image['torch.cuda.LongTensor']   = (
        nnutils_mask_image_from_size_gpu_s64)
    _mask_image['torch.cuda.FloatTensor']  = (
        nnutils_mask_image_from_size_gpu_f32)
    _mask_image['torch.cuda.DoubleTensor'] = (
        nnutils_mask_image_from_size_gpu_f64)

class mask_image_from_size(Function):
    def __init__(self, mask_value=0, inplace=False):
        super(mask_image_from_size, self).__init__()
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
            _mask_image[masked_batch.type()](x, xs, self._mask_value)
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
            _mask_image[x.type()](x, xs, 0)
            return masked_grad, None

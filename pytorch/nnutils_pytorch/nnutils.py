import torch
from torch.autograd import Function, Variable
from torch.utils.ffi import _wrap_function
from ._nnutils import lib as _lib, ffi as _ffi

_TENSOR_INT_DTYPE = ['u8', 's8', 's16', 's32', 's64']
_TENSOR_INT_SUFFIX = [
    'ByteTensor',
    'CharTensor',
    'ShortTensor',
    'IntTensor',
    'LongTensor',
]
_TENSOR_REAL_DTYPE = ['f32', 'f64']
_TENSOR_REAL_SUFFIX = [
    'FloatTensor',
    'DoubleTensor'
]
_TENSOR_ALL_DTYPE  = _TENSOR_INT_DTYPE  + _TENSOR_REAL_DTYPE
_TENSOR_ALL_SUFFIX = _TENSOR_INT_SUFFIX + _TENSOR_REAL_SUFFIX


def _import_symbols(loc):
    for symbol in dir(_lib):
        fn = getattr(_lib, symbol)
        loc[symbol] = _wrap_function(fn, _ffi)

def _create_impl_dict(loc, func_prefix, types):
    result = {}
    for tensor_suffix, data_type in types:
        cpu_fn = '%s_cpu_%s' % (func_prefix, data_type)
        gpu_fn = '%s_gpu_%s' % (func_prefix, data_type)
        if cpu_fn in loc:
            result['torch.%s' % tensor_suffix] = loc[cpu_fn]
        if gpu_fn in loc and torch.cuda.is_available():
            result['torch.cuda.%s' % tensor_suffix] = loc[gpu_fn]
    return result

# Import symbols from the dynamic library
_import_symbols(locals())

class _FunctionBase(Function):
    @classmethod
    def _assert_call(cls, func_dict, *args, **kwargs):
        # Convert variables to tensors
        converted_args = [arg.data if isinstance(arg, Variable) else arg
                          for arg in args]
        # Get tensor type.
        # If a tensor_type keyword is given, use that string. Otherwise,
        # use the type() of the first argument.
        if 'tensor_type' in kwargs:
            tensor_type = kwargs['tensor_type']
            del kwargs['tensor_type']
        else:
            tensor_type = converted_args[0].type()
        # Get function for the tensor type and call it with the args
        fn = func_dict.get(tensor_type, None)
        assert fn is not None, (
            'Class %s does not support type %s' % (
                cls.__name__, tensor_type))
        return fn(*converted_args)

_mask_image = _create_impl_dict(
    locals(), 'nnutils_mask_image_from_size',
    zip(_TENSOR_ALL_SUFFIX, _TENSOR_ALL_DTYPE))

class _MaskImageFromSize(_FunctionBase):
    @classmethod
    def forward(cls, ctx, batch_input, batch_sizes=None, mask_value=0,
                inplace=False):
        if batch_sizes is None:
            return batch_input
        else:
            ctx.save_for_backward(batch_sizes)
            ctx.mask_value = mask_value
            ctx.inplace = inplace
            assert(batch_input.is_cuda == batch_sizes.is_cuda)
            batch_output = batch_input if ctx.inplace else batch_input.clone()
            cls._assert_call(_mask_image,
                              # arguments to the actual C function
                              batch_output, batch_sizes, mask_value)
            return batch_output

    @classmethod
    def backward(cls, ctx, grad_output):
        batch_sizes, = ctx.saved_tensors
        if batch_sizes is None:
            return grad_output
        else:
            assert(grad_output.is_cuda == batch_sizes.is_cuda)
            grad_input = grad_output if ctx.inplace else grad_output.clone()
            cls._assert_call(_mask_image,
                             # arguments to the actual C function
                             grad_input, batch_sizes, 0)
            return grad_input, None, None, None


_adap_avgpool_2d_fwd = _create_impl_dict(
    locals(), 'nnutils_adaptive_avgpool_2d_fwd',
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE))
_adap_avgpool_2d_bwd = _create_impl_dict(
    locals(), 'nnutils_adaptive_avgpool_2d_bwd',
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE))

class _AdaptiveAvgpool2d(_FunctionBase):
    @classmethod
    def forward(cls, ctx, batch_input, batch_sizes, out_h, out_w):
        assert(batch_input.is_cuda == batch_sizes.is_cuda)
        ctx.save_for_backward(batch_sizes)
        N, C, inp_h, inp_w = batch_input.size()
        ctx.inp_h = inp_h
        ctx.inp_w = inp_w
        batch_output = batch_input.new(N, C, out_h, out_w)
        cls._assert_call(_adap_avgpool_2d_fwd,
                         batch_input, batch_sizes, out_h, out_w,
                         batch_output)
        return batch_output

    @classmethod
    def backward(cls, ctx, grad_output):
        batch_sizes, = ctx.saved_tensors
        assert(grad_output.is_cuda == batch_sizes.is_cuda)
        N, C, _, _ = grad_output.size()
        grad_input = grad_output.data.new(N, C, ctx.inp_h, ctx.inp_w)
        cls._assert_call(_adap_avgpool_2d_bwd,
                          grad_output, batch_sizes, grad_input)
        return Variable(grad_input, requires_grad=True), None, None, None


_adap_maxpool_2d_fwd = _create_impl_dict(
    locals(), 'nnutils_adaptive_maxpool_2d_fwd',
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE))
_adap_maxpool_2d_bwd = _create_impl_dict(
    locals(), 'nnutils_adaptive_maxpool_2d_bwd',
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE))

class _AdaptiveMaxpool2d(_FunctionBase):
    @classmethod
    def forward(cls, ctx, batch_input, batch_sizes, out_h, out_w):
        assert(batch_input.is_cuda == batch_sizes.is_cuda)
        ctx.save_for_backward(batch_sizes)
        N, C, inp_h, inp_w = batch_input.size()
        ctx.inp_h = inp_h
        ctx.inp_w = inp_w
        batch_output = batch_input.new(N, C, out_h, out_w)
        index = batch_sizes.new(N, C, out_h, out_w)
        cls._assert_call(_adap_maxpool_2d_fwd,
                         batch_input, batch_sizes, out_h, out_w,
                         batch_output, index)
        ctx.save_for_backward(index)
        return batch_output

    @classmethod
    def backward(cls, ctx, grad_output):
        batch_sizes, index = ctx.saved_tensors
        assert(grad_output.is_cuda == batch_sizes.is_cuda)
        N, C, _, _ = grad_output.size()
        grad_input = grad_output.data.new(N, C, ctx.inp_h, ctx.inp_w)
        cls._assert_call(_adap_maxpool_2d_bwd,
                          grad_output, batch_sizes, index, grad_input)
        return Variable(grad_input, requires_grad=True), None, None, None


def mask_image_from_size(batch_input, batch_sizes=None, mask_value=0,
                         inplace=False):
    return _MaskImageFromSize.apply(batch_input, batch_sizes, mask_value,
                                    inplace)

def adaptive_avgpool_2d(batch_input, batch_sizes, out_h, out_w):
    return _AdaptiveAvgpool2d.apply(batch_input, batch_sizes, out_h, out_w)

def adaptive_maxpool_2d(batch_input, batch_sizes, out_h, out_w):
    return _AdaptiveMaxpool2d.apply(batch_input, batch_sizes, out_h, out_w)

__all__ = [mask_image_from_size, adaptive_avgpool_2d, adaptive_maxpool_2d]

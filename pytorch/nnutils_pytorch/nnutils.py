import torch
from torch.autograd import Function, Variable
from ._nnutils import *

_TENSOR_INT_DTYPE = ["u8", "s8", "s16", "s32", "s64"]
_TENSOR_INT_SUFFIX = [
    "ByteTensor",
    "CharTensor",
    "ShortTensor",
    "IntTensor",
    "LongTensor",
]
_TENSOR_REAL_DTYPE = ["f32", "f64"]
_TENSOR_REAL_SUFFIX = ["FloatTensor", "DoubleTensor"]
_TENSOR_ALL_DTYPE = _TENSOR_INT_DTYPE + _TENSOR_REAL_DTYPE
_TENSOR_ALL_SUFFIX = _TENSOR_INT_SUFFIX + _TENSOR_REAL_SUFFIX


def _create_impl_dict(loc, func_prefix, types):
    result = {}
    for tensor_suffix, data_type in types:
        cpu_fn = "%s_cpu_%s" % (func_prefix, data_type)
        gpu_fn = "%s_gpu_%s" % (func_prefix, data_type)
        if cpu_fn in loc:
            result["torch.%s" % tensor_suffix] = loc[cpu_fn]
        if gpu_fn in loc and torch.cuda.is_available():
            result["torch.cuda.%s" % tensor_suffix] = loc[gpu_fn]
    return result


class _FunctionBase(Function):
    @classmethod
    def _assert_call(cls, func_dict, *args, **kwargs):
        # Convert variables to tensors
        converted_args = [
            arg.data if isinstance(arg, Variable) else arg for arg in args
        ]
        # Get tensor type.
        # If a tensor_type keyword is given, use that string. Otherwise,
        # use the type() of the first argument.
        if "tensor_type" in kwargs:
            tensor_type = kwargs["tensor_type"]
            del kwargs["tensor_type"]
        else:
            tensor_type = converted_args[0].type()
        # Get function for the tensor type and call it with the args
        fn = func_dict.get(tensor_type, None)
        assert fn is not None, "Class %s does not support type %s" % (
            cls.__name__,
            tensor_type,
        )
        return fn(*converted_args)


_mask_image = _create_impl_dict(
    locals(), "nnutils_mask_image_from_size", zip(_TENSOR_ALL_SUFFIX, _TENSOR_ALL_DTYPE)
)


class _MaskImageFromSize(_FunctionBase):
    @classmethod
    def forward(cls, ctx, batch_input, batch_sizes=None, mask_value=0, inplace=False):
        if batch_sizes is None:
            return batch_input
        else:
            batch_input = batch_input.contiguous()
            batch_sizes = batch_sizes.contiguous() if batch_sizes is not None else None
            ctx.save_for_backward(batch_sizes)
            ctx.mask_value = mask_value
            ctx.inplace = inplace
            assert batch_input.is_cuda == batch_sizes.is_cuda
            batch_output = batch_input if ctx.inplace else batch_input.clone()
            cls._assert_call(
                _mask_image,
                # arguments to the actual C function
                batch_output,
                batch_sizes,
                mask_value,
            )
            return batch_output

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_output = grad_output.contiguous()
        batch_sizes, = ctx.saved_tensors
        if batch_sizes is None:
            return grad_output
        else:
            assert grad_output.is_cuda == batch_sizes.is_cuda
            grad_input = grad_output if ctx.inplace else grad_output.clone()
            cls._assert_call(
                _mask_image,
                # arguments to the actual C function
                grad_input,
                batch_sizes,
                # Note: Gradient in the masked areas is 0, not "masked_value"!
                0,
            )
            return grad_input, None, None, None


_adap_avgpool_2d_fwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_avgpool_2d_fwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)
_adap_avgpool_2d_bwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_avgpool_2d_bwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)

_adap_avgpool_2d_generic_fwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_avgpool_2d_generic_fwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)
_adap_avgpool_2d_generic_bwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_avgpool_2d_generic_bwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)


class _AdaptiveAvgpool2d(_FunctionBase):
    @classmethod
    def forward(cls, ctx, batch_input, batch_sizes, out_h, out_w):
        assert out_h is not None or out_w is not None
        assert batch_input.is_cuda == batch_sizes.is_cuda
        ctx.save_for_backward(batch_sizes)
        batch_input = batch_input.contiguous()
        batch_sizes = batch_sizes.contiguous()
        N, C, inp_h, inp_w = batch_input.size()
        ctx.inp_h = inp_h
        ctx.inp_w = inp_w
        ctx.out_h = out_h
        ctx.out_w = out_w

        if out_h is None or out_w is None:
            output_sizes = batch_sizes.clone()
            if out_h is not None:
                output_sizes[:, 0] = out_h
            if out_w is not None:
                output_sizes[:, 1] = out_w
            out_h = inp_h if out_h is None else out_h
            out_w = inp_w if out_w is None else out_w

            batch_output = batch_input.new(N, C, out_h, out_w).zero_()
            cls._assert_call(
                _adap_avgpool_2d_generic_fwd,
                batch_sizes,
                output_sizes,
                batch_input,
                batch_output,
                tensor_type=batch_input.type(),
            )
            ctx.output_sizes = output_sizes
        else:
            batch_output = batch_input.new(N, C, out_h, out_w).zero_()
            cls._assert_call(
                _adap_avgpool_2d_fwd,
                batch_sizes,
                batch_input,
                batch_output,
                tensor_type=batch_input.type(),
            )
        return batch_output

    @classmethod
    def backward(cls, ctx, grad_output):
        batch_sizes, = ctx.saved_tensors
        assert grad_output.is_cuda == batch_sizes.is_cuda
        grad_output = grad_output.contiguous()
        N, C, _, _ = grad_output.size()
        grad_input = grad_output.data.new(N, C, ctx.inp_h, ctx.inp_w).zero_()
        if ctx.out_h is None or ctx.out_w is None:
            cls._assert_call(
                _adap_avgpool_2d_generic_bwd,
                batch_sizes,
                ctx.output_sizes,
                grad_output,
                grad_input,
                tensor_type=grad_output.data.type(),
            )
        else:
            cls._assert_call(
                _adap_avgpool_2d_bwd,
                batch_sizes,
                grad_output,
                grad_input,
                tensor_type=grad_output.data.type(),
            )
        return Variable(grad_input), None, None, None


_adap_maxpool_2d_fwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_maxpool_2d_fwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)
_adap_maxpool_2d_bwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_maxpool_2d_bwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)

_adap_maxpool_2d_generic_fwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_maxpool_2d_generic_fwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)
_adap_maxpool_2d_generic_bwd = _create_impl_dict(
    locals(),
    "nnutils_adaptive_maxpool_2d_generic_bwd",
    zip(_TENSOR_REAL_SUFFIX, _TENSOR_REAL_DTYPE),
)


class _AdaptiveMaxpool2d(_FunctionBase):
    @classmethod
    def forward(cls, ctx, batch_input, batch_sizes, out_h, out_w):
        assert out_h is not None or out_w is not None
        assert batch_input.is_cuda == batch_sizes.is_cuda
        batch_input = batch_input.contiguous()
        batch_sizes = batch_sizes.contiguous()
        N, C, inp_h, inp_w = batch_input.size()
        ctx.inp_h = inp_h
        ctx.inp_w = inp_w
        ctx.out_h = out_h
        ctx.out_w = out_w
        if out_h is None or out_w is None:
            output_sizes = batch_sizes.clone()
            if out_h is not None:
                output_sizes[:, 0] = out_h
            if out_w is not None:
                output_sizes[:, 1] = out_w
            out_h = inp_h if out_h is None else out_h
            out_w = inp_w if out_w is None else out_w

            batch_output = batch_input.new(N, C, out_h, out_w).zero_()
            index = batch_sizes.new(N, C, out_h, out_w)
            cls._assert_call(
                _adap_maxpool_2d_generic_fwd,
                batch_sizes,
                output_sizes,
                batch_input,
                batch_output,
                index,
                tensor_type=batch_input.type(),
            )
            ctx.output_sizes = output_sizes
        else:
            batch_output = batch_input.new(N, C, out_h, out_w).zero_()
            index = batch_sizes.new(N, C, out_h, out_w)
            cls._assert_call(
                _adap_maxpool_2d_fwd,
                batch_sizes,
                batch_input,
                batch_output,
                index,
                tensor_type=batch_input.type(),
            )
        ctx.save_for_backward(index)
        return batch_output, index

    @classmethod
    def backward(cls, ctx, grad_output, unused_grad_indexes):
        index, = ctx.saved_tensors
        assert grad_output.is_cuda == index.is_cuda
        assert grad_output.size() == index.size()
        grad_output = grad_output.contiguous()
        N, C, _, _ = grad_output.size()
        grad_input = grad_output.data.new(N, C, ctx.inp_h, ctx.inp_w).zero_()
        if ctx.out_h is None or ctx.out_w is None:
            cls._assert_call(
                _adap_maxpool_2d_generic_bwd,
                ctx.output_sizes,
                index,
                grad_output,
                grad_input,
                tensor_type=grad_output.data.type(),
            )
        else:
            cls._assert_call(
                _adap_maxpool_2d_bwd,
                index,
                grad_output,
                grad_input,
                tensor_type=grad_output.data.type(),
            )
        return Variable(grad_input), None, None, None


def mask_image_from_size(batch_input, batch_sizes=None, mask_value=0, inplace=False):
    return _MaskImageFromSize.apply(batch_input, batch_sizes, mask_value, inplace)


def adaptive_avgpool_2d(batch_input, output_sizes, batch_sizes=None):
    r"""Applies a 2D adaptive average pooling over an input signal composed of
    several input planes.

    You may specify a single dimension for pooling, in that case the other
    dimension will keep its original size.

    If your input is composed of multiple padded images with different sizes,
    you can do the pooling taking into account the original size of each image
    in the batch, by using the batch_sizes argument.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple). One of the two integers may be None, to
            keep the original size in that dimension.
        batch_sizes: a N x 2 matrix containing the size of each image in the
            batch. Default: ``None``
    """
    if batch_sizes is None:
        return torch.nn.functional.adaptive_avg_pool2d(batch_input, output_sizes)
    else:
        if isinstance(output_sizes, (list, tuple)):
            out_h, out_w = output_sizes
        else:
            out_h, out_w = output_sizes, output_sizes
        return _AdaptiveAvgpool2d.apply(batch_input, batch_sizes, out_h, out_w)


def adaptive_maxpool_2d(
    batch_input, output_sizes, batch_sizes=None, return_indices=False
):
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    You may specify a single dimension for pooling, in that case the other
    dimension will keep its original size.

    If your input is composed of multiple padded images with different sizes,
    you can do the pooling taking into account the original size of each image
    in the batch, by using the batch_sizes argument.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple). One of the two integers may be None, to
            keep the original size in that dimension.
        batch_sizes: a N x 2 matrix containing the size of each image in the
            batch. Default: ``None``
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if batch_sizes is None:
        return torch.nn.functional.adaptive_max_pool2d(
            batch_input, output_sizes, return_indices
        )
    else:
        if isinstance(output_sizes, (list, tuple)):
            out_h, out_w = output_sizes
        else:
            out_h, out_w = output_sizes, output_sizes
        ret = _AdaptiveMaxpool2d.apply(batch_input, batch_sizes, out_h, out_w)
        return ret if return_indices else ret[0]


_gpu_fn = locals().get("nnutils_mask_image_from_size_gpu_f32", None)


def is_cuda_available():
    return torch.cuda.is_available() and _gpu_fn is not None


__all__ = [
    "is_cuda_available",
    "mask_image_from_size",
    "adaptive_avgpool_2d",
    "adaptive_maxpool_2d",
]

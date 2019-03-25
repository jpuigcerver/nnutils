import torch

import nnutils_pytorch._C as _nnutils_pytorch


class _AdaptiveAvgpool2d(torch.autograd.Function):
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

        if out_h is None or out_w is None:
            output_sizes = batch_sizes.clone()
            if out_h is not None:
                output_sizes[:, 0] = out_h
            if out_w is not None:
                output_sizes[:, 1] = out_w
            out_h = inp_h if out_h is None else out_h
            out_w = inp_w if out_w is None else out_w
        else:
            output_sizes = None

        ctx.output_sizes = output_sizes
        batch_output = batch_input.new(N, C, out_h, out_w).zero_()
        _nnutils_pytorch.adaptive_avgpool_2d_fwd(
            x=batch_input, y=batch_output, xs=batch_sizes, ys=output_sizes
        )
        return batch_output

    @classmethod
    def backward(cls, ctx, grad_output):
        batch_sizes, = ctx.saved_tensors
        assert grad_output.is_cuda == batch_sizes.is_cuda
        grad_output = grad_output.contiguous()
        N, C, _, _ = grad_output.size()
        grad_input = grad_output.data.new(N, C, ctx.inp_h, ctx.inp_w).zero_()
        _nnutils_pytorch.adaptive_avgpool_2d_bwd(
            grad_y=grad_output, grad_x=grad_input, xs=batch_sizes, ys=ctx.output_sizes
        )
        return grad_input, None, None, None


class _AdaptiveMaxpool2d(torch.autograd.Function):
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
        else:
            output_sizes = None

        batch_output = batch_input.new(N, C, out_h, out_w).zero_()
        index = batch_sizes.new(N, C, out_h, out_w)
        _nnutils_pytorch.adaptive_maxpool_2d_fwd(
            x=batch_input, y=batch_output, index=index, xs=batch_sizes, ys=output_sizes
        )
        ctx.output_sizes = output_sizes
        ctx.save_for_backward(index)
        return batch_output, index

    @classmethod
    def backward(cls, ctx, grad_output, unused_grad_indexes):
        index, = ctx.saved_tensors
        assert grad_output.is_cuda == index.is_cuda
        assert grad_output.size() == index.size()
        grad_output = grad_output.contiguous()
        N, C, _, _ = grad_output.size()
        grad_input = grad_output.new(N, C, ctx.inp_h, ctx.inp_w).zero_()
        _nnutils_pytorch.adaptive_maxpool_2d_bwd(
            grad_y=grad_output, grad_x=grad_input, index=index, ys=ctx.output_sizes
        )
        return grad_input, None, None, None


class _MaskImageFromSizeFunction(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, batch_input, batch_sizes, mask_value=0, inplace=False):
        ctx.save_for_backward(batch_sizes)
        ctx.mask_value = mask_value
        ctx.inplace = inplace
        batch_output = batch_input if ctx.inplace else batch_input.clone()
        _nnutils_pytorch.mask_image_from_size(
            x=batch_output.contiguous(), xs=batch_sizes.contiguous(), mask=mask_value
        )
        return batch_output

    @classmethod
    def backward(cls, ctx, grad_output):
        batch_sizes, = ctx.saved_tensors
        grad_input = grad_output if ctx.inplace else grad_output.clone()
        _nnutils_pytorch.mask_image_from_size(
            x=grad_input.contiguous(), xs=batch_sizes.contiguous(), mask=0
        )
        return grad_input, None, None, None


def adaptive_avgpool_2d(batch_input, output_sizes, batch_sizes=None):
    """Applies a 2D adaptive average pooling over an input signal composed of
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
    """Applies a 2D adaptive max pooling over an input signal composed of
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


def mask_image_from_size(batch_input, batch_sizes=None, mask_value=0, inplace=False):
    """Mask a batch of images (a 4D tensor) from each image size.

    Let (h, w) be the height and width of a given image in the batch, then this
    function sets the value of all pixel coordinates with x >= w or y >= h to
    the value given by the argument ``mask_value''.

    Args:
        batch_input: the input batch (a 4D tensor), with layout N x C x H x W
            where N is the number of images in the batch, C is the number of
            channels of each image, H is the maximum height and W is the
            maximum width.
        batch_sizes: a integer matrix containing the height and width of each
            image in the batch (N x 2 matrix). If None, does not perform any
            masking.
        mask_value: value used for the pixels ``outside'' of the image bounding
            box. Default: 0
        inplace: whether to perform the operation inplace or not.
            Default: ``False``
    """
    if batch_sizes is None:
        return batch_input
    else:
        return _MaskImageFromSizeFunction.apply(
            batch_input, batch_sizes, mask_value, inplace
        )

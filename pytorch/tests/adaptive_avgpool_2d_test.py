import pytest
import torch
from torch.nn.functional import adaptive_avg_pool2d as torch_adaptive_avg_pool2d

from nnutils_pytorch import adaptive_avgpool_2d


def prepare(dtype, device):
    return {
        "s": torch.tensor([[3, 4], [2, 8]], dtype=torch.long, device=device),
        "x": (
            torch.tensor(
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
                ],
                dtype=dtype,
                device=device,
            )
            .resize_(2, 1, 3, 8)
            .requires_grad_()
        ),
        "dy": torch.tensor(
            [
                # Output gradient w.r.t Image 1
                [[3, 6, 9, 12]],
                # Output gradient w.r.t. Image 2
                [[8, 12, 16, 20]],
            ],
            dtype=dtype,
            device=device,
        ).resize_(2, 1, 1, 4),
        "dy_fixed_height": torch.tensor(
            [
                # Output gradient w.r.t. Image 1
                [[3, 6, 9, 12, 0, 0, 0, 0]],
                # Output gradient w.r.t. Image 2
                [[6, 8, 10, 12, 14, 16, 18, 20]],
            ],
            dtype=dtype,
            device=device,
        ).resize_(2, 1, 1, 8),
        "dy_fixed_width": torch.tensor(
            [
                # Output gradient w.r.t. Image 1
                [[2, 4], [6, 8], [10, 12]],
                # Output gradient w.r.t. Image 2
                [[4, 8], [12, 16], [0, 0]],
            ],
            dtype=dtype,
            device=device,
        ).resize_(2, 1, 3, 2),
        "expect_y": torch.tensor(
            [
                # Expected output 1
                [[5, 6, 7, 8]],
                # Expected output 2
                [[5.5, 7.5, 9.5, 11.5]],
            ],
            dtype=dtype,
        ).resize_(2, 1, 1, 4),
        "expect_y_fixed_height": torch.tensor(
            [
                # Expected output 1
                [[5, 6, 7, 8, 0, 0, 0, 0]],
                # Expected output 2
                [[5, 6, 7, 8, 9, 10, 11, 12]],
            ],
            dtype=dtype,
        ).resize_(2, 1, 1, 8),
        "expect_y_fixed_width": torch.tensor(
            [
                # Expected output 1
                [[1.5, 3.5], [5.5, 7.5], [9.5, 11.5]],
                # Expected output 2
                [[2.5, 6.5], [10.5, 14.5], [0, 0]],
            ],
            dtype=dtype,
        ).resize_(2, 1, 3, 2),
        "expect_dx": torch.tensor(
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
            ],
            dtype=dtype,
        ).resize_(2, 1, 3, 8),
        "expect_dx_fixed_height": torch.tensor(
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
            ],
            dtype=dtype,
        ).resize_(2, 1, 3, 8),
        "expect_dx_fixed_width": torch.tensor(
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
            ],
            dtype=dtype,
        ).resize_(2, 1, 3, 8),
    }


dtypes = [
    pytest.param(
        torch.half,
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"),
    ),
    torch.float,
    torch.double,
]
devices = [
    pytest.param(
        torch.device("cuda"),
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA"),
    ),
    torch.device("cpu"),
]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("device", devices)
def test_base(dtype, device):
    t = prepare(dtype, device)
    x, xs = t["x"], t["s"]
    y = adaptive_avgpool_2d(x, output_sizes=(1, 4), batch_sizes=xs)
    y.backward(t["dy"], retain_graph=True)
    torch.testing.assert_allclose(y, t["expect_y"])
    torch.testing.assert_allclose(x.grad, t["expect_dx"])


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("dim", ["height", "width"])
def test_fixed(dtype, device, dim):
    t = prepare(dtype, device)
    x, xs = t["x"], t["s"]
    output_sizes = (1, None) if dim == "height" else (None, 2)
    y = adaptive_avgpool_2d(x, output_sizes=output_sizes, batch_sizes=xs)
    y.backward(t[f"dy_fixed_{dim}"], retain_graph=True)
    torch.testing.assert_allclose(y, t[f"expect_y_fixed_{dim}"])
    torch.testing.assert_allclose(x.grad, t[f"expect_dx_fixed_{dim}"])


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("size", [(2, 3), (20, 25)])
def test_compare_reference(dtype, device, size):
    x3 = torch.randn(2, 3, 10, 15, dtype=dtype, device=device, requires_grad=True)
    xs3 = torch.tensor([[4, 5], [8, 6]], dtype=torch.long, device=device)
    x1 = x3[0, :, :4, :5].clone().view(1, 3, 4, 5)
    x2 = x3[1, :, :8, :6].clone().view(1, 3, 8, 6)
    # Compare forward
    y1 = torch_adaptive_avg_pool2d(x1, output_size=size)
    y2 = torch_adaptive_avg_pool2d(x2, output_size=size)
    y3 = adaptive_avgpool_2d(x3, output_sizes=size, batch_sizes=xs3)
    torch.testing.assert_allclose(y3, torch.cat([y1, y2]))
    # Compare backward
    dx1, dx2 = torch.autograd.grad(y1.sum() + y2.sum(), [x1, x2])
    (dx3,) = torch.autograd.grad(y3.sum(), [x3])
    ref = torch.zeros_like(dx3)
    ref[0, :, :4, :5] = dx1
    ref[1, :, :8, :6] = dx2
    torch.testing.assert_allclose(dx3, ref)

import pytest
import torch

from nnutils_pytorch import mask_image_from_size


@pytest.mark.parametrize(
    "dtype",
    [
        torch.uint8,
        torch.short,
        torch.int,
        torch.long,
        pytest.param(
            torch.half,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        torch.float,
        torch.double,
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="Requires CUDA"
            ),
        ),
        torch.device("cpu"),
    ],
)
def test_base(dtype, device):
    x = torch.randn(5, 3, 213, 217, device=device).requires_grad_()
    xs = torch.tensor(
        [[29, 29], [213, 217], [213, 15], [15, 217], [97, 141]],
        dtype=torch.long,
        device=device,
    )
    # Cost for each output pixel
    cy = torch.randn(5, 3, 213, 217, device=device).to(dtype)
    y = mask_image_from_size(batch_input=x, batch_sizes=xs, mask_value=99)

    # Check forward
    for i, (xi, yi, s) in enumerate(zip(x, y, xs)):
        # Check non-masked area
        d = torch.sum(yi[:, : s[0], : s[1]] != xi[:, : s[0], : s[1]])
        assert d == 0, f"Sample {i} failed in the non-masked area"
        # Check masked area
        d1 = torch.sum(yi[:, s[0] :, :] != 99) if s[0] < x.size(2) else 0
        d2 = torch.sum(yi[:, :, s[1] :] != 99) if s[1] < x.size(3) else 0
        assert d1 + d2 == 0, f"Sample {i} failed in the masked area"

    # Check gradients
    cost = torch.sum(torch.mul(y, cy))
    (dx,) = torch.autograd.grad(cost, (x,))
    for i, (xi, yi, s) in enumerate(zip(dx, cy, xs)):
        # Check non-masked area
        d = torch.sum(xi[:, : s[0], : s[1]] != yi[:, : s[0], : s[1]])
        assert d == 0, f"Sample {i} failed in the non-masked area"
        # Check masked area
        d1 = torch.sum(xi[:, s[0] :, :] != 0) if s[0] < x.size(2) else 0
        d2 = torch.sum(xi[:, :, s[1] :] != 0) if s[1] < x.size(3) else 0
        assert d1 + d2 == 0, f"Sample {i} failed in the masked area"

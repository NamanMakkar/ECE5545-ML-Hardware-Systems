import torch
import pytest
import numpy as np
from src.conv2d import conv2d
import torch.nn.functional as F


################
# Conv2D Tests #
################
CONV2D_MAX_INPUT_SIZE = 50
CONV2D_NUM_RUNS = 100


@pytest.mark.parametrize('method_eps', [
    ('naive', 1e-4),
    ('im2col', 1e-4),
    ('fft', 1e-2)
])
@pytest.mark.parametrize('input_size', list(range(3, CONV2D_MAX_INPUT_SIZE, 4)))
def test_conv2d_largek(method_eps, input_size):
    precision = torch.float32
    method, eps = method_eps
    def round_up_to_odd(f):
        return int(np.ceil(f) // 2 * 2 + 1)

    loss_lst = []
    for seed in range(CONV2D_NUM_RUNS):
        torch.random.manual_seed(seed)
        input_size = round_up_to_odd(input_size)

        x = torch.randn((input_size, input_size), dtype=precision)
        k = torch.randn(size=(
            round_up_to_odd(input_size // 2),
            round_up_to_odd(input_size // 2)),
            dtype=precision)
        b = torch.randn(size=(1,), dtype=precision)

        ans = conv2d(x, k, b, method='torch').float()
        out = conv2d(x, k, b, method=method).float()
        assert ans.shape == out.shape, \
            "Shape mismatch. expected=%s output=%s" % (str(ans.shape), str(out.shape))
        loss_lst.append(F.l1_loss(ans, out).item())

    loss_avg = np.array(loss_lst).mean()
    assert loss_avg < eps, "Method:%s L1-error: %.4e (eps:%.2e)" \
                           % (method, loss_avg, eps)


@pytest.mark.parametrize('method_eps', [
    ('naive', 1e-4),
    ('im2col', 1e-4),
    ('winograd', 1e-3),
    ('fft', 1e-2)
])
@pytest.mark.parametrize('input_size', list(range(3, CONV2D_MAX_INPUT_SIZE, 4)))
def test_conv2d_3x3k(method_eps, input_size):
    precision = torch.float32
    method, eps = method_eps
    def round_up_to_odd(f):
        return int(np.ceil(f) // 2 * 2 + 1)

    loss_lst = []
    for seed in range(CONV2D_NUM_RUNS):
        torch.random.manual_seed(seed)
        input_size = round_up_to_odd(input_size)

        x = torch.randn((input_size, input_size), dtype=precision)
        k = torch.randn(size=(3, 3), dtype=precision)
        b = torch.randn(size=(1,), dtype=precision)

        ans = conv2d(x, k, b, method='torch').float()
        out = conv2d(x, k, b, method=method).float()
        assert ans.shape == out.shape, \
            "Shape mismatch. expected=%s output=%s" % (str(ans.shape), str(out.shape))
        loss_lst.append(F.l1_loss(ans, out).item())

    loss_avg = np.array(loss_lst).mean()
    assert loss_avg < eps, "Method:%s L1-error: %.4e (eps:%.2e)" \
                           % (method, loss_avg, eps)
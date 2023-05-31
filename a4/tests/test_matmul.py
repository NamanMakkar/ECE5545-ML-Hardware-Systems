import torch
import pytest
import numpy as np
from src.matmul import matmul
import torch.nn.functional as F


MATMUL_MAX_RUNS = 100
MATMUL_LOG_EPS = 1e-5
@pytest.mark.parametrize('M', (2 ** np.arange(1, 8, 2)).astype(np.int32))
@pytest.mark.parametrize('N', (2 ** np.arange(1, 8, 2)).astype(np.int32))
@pytest.mark.parametrize('K', (2 ** np.arange(1, 8, 2)).astype(np.int32))
@pytest.mark.parametrize('precision', [torch.float32])
def test_logmatmul(M, N, K, precision):
    losses = []
    for seed in range(MATMUL_MAX_RUNS):
        torch.random.manual_seed(seed)
        A = torch.randn(M, N, dtype=precision)
        B = torch.randn(N, K, dtype=precision)
        ans = matmul(A, B, method='torch')
        out = matmul(A, B, method='log')
        assert ans.shape == out.shape, \
            "Shape mismatch. expected=%s output=%s" \
            % (str(ans.shape), str(out.shape))
        losses.append(F.l1_loss(ans, out).item())
    loss_avg = np.array(losses).mean()
    assert loss_avg < MATMUL_LOG_EPS, "SEED: %d L1: %.4e (eps:%.2e)" \
                                      % (seed, loss_avg, MATMUL_LOG_EPS)

mnist_nn_w = torch.load("data/mnist_fc.pt")
mnist_nn_a = torch.load("data/mnist_act.pt")
@pytest.mark.parametrize('key_rank_eps', [
    ('fc1.weight', "act.0", None, None, 1e-6),
    ('fc1.weight', "act.0", None, 128,  1e-1),
    ('fc1.weight', "act.0", None, 16,   3.),
    ('fc1.weight', "act.0", 256, None,  1e-2),
    ('fc1.weight', "act.0", 128, None,  1e-1),
    ('fc1.weight', "act.0", 64, None,   1e-1),
    ('fc1.weight', "act.0", 16, None,   2.),

    ('fc2.weight', "act.1", None, None, 1e-6),
    ('fc2.weight', "act.1", None, 128,  1e-6),
    ('fc2.weight', "act.1", None, 64,   2e-1),
    ('fc2.weight', "act.1", None, 16,   1.),
    ('fc2.weight', "act.1", 128, None,  5e-2),
    ('fc2.weight', "act.1", 64, None,   2e-1),
    ('fc2.weight', "act.1", 16, None,   2.),

    ('fc3.weight', "act.2", None, None, 1e-6),
    ('fc3.weight', "act.2", None, 10,   1e-6),
    ('fc3.weight', "act.2", None, 8,    10),
    ('fc3.weight', "act.2", 128, None,  1e-6),
    ('fc3.weight', "act.2", 64, None,   5e-2),
    ('fc3.weight', "act.2", 16, None,   5e-1),
])
@pytest.mark.parametrize('precision', [torch.float32])
def test_svd(key_rank_eps, precision):
    keyw, keya, rank_A, rank_B, eps = key_rank_eps
    A = mnist_nn_a[keya].to(precision)
    B = mnist_nn_w[keyw].to(precision).transpose(0, 1)
    ans = matmul(A, B, method='torch')
    out = matmul(A, B, method='svd', rank_A=rank_A, rank_B=rank_B)
    assert ans.shape == out.shape, \
        "Shape mismatch. expected=%s output=%s" \
        % (str(ans.shape), str(out.shape))
    loss = F.mse_loss(ans, out).item()
    print(loss)
    assert loss < eps, "Key: (%s, %s) RANK:(%s, %s) MSE: %.4e (eps:%.2e)" \
                       % (keyw, keya, rank_A, rank_B, loss, eps)

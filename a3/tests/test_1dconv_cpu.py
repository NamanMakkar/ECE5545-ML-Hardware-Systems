import tvm
import torch
import pytest
import timeit
import numpy as np
import torch.nn.functional as F
from src.ops import make_conv1d_cpu_scheduler


dev = tvm.device('llvm', 0)


def make_conv1d_cpu_func(M, N):
    s, A, W, O = make_conv1d_cpu_scheduler(M, N)
    func = tvm.build(s, [A, W, O], "llvm")
    return func


def ans_np(a_np, w_np):
    a_np = a_np.flatten()
    w_np = w_np.flatten()
    return np.convolve(a_np, w_np)


def ans_torch(a_torch, w_torch):
    M, N = a_torch.size(0), w_torch.size(0)
    b_torch = F.conv1d(a_torch, w_torch, bias=None, stride=1,
                       padding=(N - 1), dilation=1, groups=1)
    return b_torch


@pytest.mark.parametrize('execution_number', range(5))
def test1_M1_N1(execution_number):
    # Define dimension
    M = 1
    N = 1
    func = make_conv1d_cpu_func(M, N)

    # Create random test data
    np.random.seed(seed=execution_number)
    a_np = np.random.rand(M).astype(np.float32)
    w_np = np.random.rand(N).astype(np.float32)
    b_np = ans_np(a_np, w_np)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M + N - 1), dtype='float32'), dev)
    func(a, w, b)
    b_out = b.numpy()

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


@pytest.mark.parametrize('execution_number', [1, 10, 100, 1000, 10000])
def test1_Mvar_N1024(execution_number):
    # Define dimension
    M = execution_number
    N = 1024
    func = make_conv1d_cpu_func(M, N)

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M).astype(np.float32)
    w_np = np.random.rand(N).astype(np.float32)
    b_np = ans_np(a_np, w_np)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M + N - 1), dtype='float32'), dev)
    func(a, w, b)
    b_out = b.numpy()

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


@pytest.mark.parametrize('execution_number', [1, 10, 100, 1000, 10000])
def test1_M1024_Nvar(execution_number):
    # Define dimension
    M = 1024
    N = execution_number
    func = make_conv1d_cpu_func(M, N)

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M).astype(np.float32)
    w_np = np.random.rand(N).astype(np.float32)
    b_np = ans_np(a_np, w_np)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M + N - 1), dtype='float32'), dev)
    func(a, w, b)
    b_out = b.numpy()

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)

"""
@pytest.mark.parametrize(
    'execution_number', [2, 4, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
def test1_speed_torch(execution_number):
    # Define dimension
    M = 1024
    N = 1024
    n_repeat = 100

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M).astype(np.float32)
    w_np = np.random.rand(N).astype(np.float32)

    # Torch input
    a_torch = torch.tensor([[a_np]]).float()
    w_torch = torch.tensor([[np.flip(w_np)]]).float()
    if torch.cuda.is_available():
        a_torch = a_torch.cuda()
        w_torch = w_torch.cuda()

    # Time the torch implementation
    def torch_time():
        ans_torch(a_torch, w_torch)
    time_torch = timeit.timeit(torch_time, number=n_repeat)

    # Time the optimized implementation
    func = make_conv1d_cpu_func(M, N)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M + N - 1), dtype='float32'), dev)
    func(a, w, b)
    def tvm_time():
        func(a, w, b)
    time_tvm = timeit.timeit(tvm_time, number=n_repeat)

    opt_folds = float(execution_number)
    assert time_tvm * opt_folds <= time_torch, \
        "%dx speed-up failed: TVM Time: %.5es TorchTime: %.5es" \
        % (execution_number, time_tvm, time_torch, )
"""

import tvm
import torch
import pytest
import timeit
import numpy as np
from src.ops import make_gemm_gpu_scheduler


dev = tvm.cuda(0)

def ans_np(a, b):
    return np.matmul(a, b)


def make_func(M, K, N):
    s, A, B, O = make_gemm_gpu_scheduler(M, K, N)
    func = tvm.build(s, [A, B, O], "cuda")
    return func


def ans_torch(a_torch, b_torch):
    torch.cuda.synchronize()
    out = torch.mm(a_torch, b_torch)
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize('execution_number', range(5))
def test1_M1_N1_K2(execution_number):
    # Define dimension
    M = 1
    N = 1
    K = 2
    func = make_func(M, K, N)

    # Create random test data
    np.random.seed(seed=execution_number)
    a_np = np.random.rand(M, K).astype(np.float32)
    w_np = np.random.rand(K, N).astype(np.float32)
    b_np = ans_np(a_np, w_np)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M, N), dtype='float32'), dev)
    func(a, w, b)
    b_out = b.numpy()

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


@pytest.mark.parametrize('execution_number', [1, 10, 100, 1000, 10000])
def test1_Mvar_N1024_K1024(execution_number):
    # Define dimension
    M = execution_number
    N = 1024
    K = 1024
    func = make_func(M, K, N)

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M, K).astype(np.float32)
    w_np = np.random.rand(K, N).astype(np.float32)
    b_np = ans_np(a_np, w_np)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M, N), dtype='float32'), dev)
    func(a, w, b)
    b_out = b.numpy()

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


@pytest.mark.parametrize('execution_number', [1, 10, 100, 1000, 10000])
def test1_M1024_Nvar_K1024(execution_number):
    # Define dimension
    M = 1024
    N = execution_number
    K = 1024
    func = make_func(M, K, N)

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M, K).astype(np.float32)
    w_np = np.random.rand(K, N).astype(np.float32)
    b_np = ans_np(a_np, w_np)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M, N), dtype='float32'), dev)
    func(a, w, b)
    b_out = b.numpy()

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


@pytest.mark.parametrize('execution_number', [1, 10, 100, 1000, 10000])
def test1_M1024_N1024_Kvar(execution_number):
    # Define dimension
    M = 1024
    N = 1024
    K = execution_number
    func = make_func(M, K, N)

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M, K).astype(np.float32)
    w_np = np.random.rand(K, N).astype(np.float32)
    b_np = ans_np(a_np, w_np)

    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M, N), dtype='float32'), dev)
    func(a, w, b)
    b_out = b.numpy()

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


"""
@pytest.mark.parametrize(
    'execution_number', [2, 10, 100, 1000, 2000, 4000, 6000, 8000, 10000])
def test1_speed_torch(execution_number):
    # Define dimension
    M = 1024
    K = 1024
    N = 1024
    n_repeat = 100

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M, K).astype(np.float32)
    w_np = np.random.rand(K, N).astype(np.float32)

    # Torch input
    a_torch = torch.tensor(a_np).float()
    w_torch = torch.tensor(w_np).float()

    # Time the torch implementation
    def torch_time():
        ans_torch(a_torch, w_torch)
    time_torch = timeit.timeit(torch_time, number=n_repeat)

    # Time the optimized implementation
    func = make_func(M, K, N)
    a = tvm.nd.array(a_np, dev)
    w = tvm.nd.array(w_np, dev)
    b = tvm.nd.array(np.zeros((M, N), dtype='float32'), dev)
    func(a, w, b)
    def tvm_time():
        func(a, w, b)
    time_tvm = timeit.timeit(tvm_time, number=n_repeat)

    opt_folds = float(execution_number)
    assert time_tvm * opt_folds <= time_torch, \
        "%dx speed-up failed: TVM Time: %.5es TorchTime: %.5es" \
        % (execution_number, time_tvm, time_torch, )
"""

# Add TVM to the Python path
import sys
sys.path.append('/tvm/python')
sys.path.append('/tvm/topi/python')
sys.path.append('/tvm/nnvm/python')
sys.path.append('/tvm/vta/python')

import vta
import pytest
import numpy as np
from src.ops_fpga import make_conv1d_fpga_function, make_conv1d_fpga_scheduler

def ans_np(a_np, w_np):
    a_np = a_np.flatten()
    w_np = w_np.flatten()
    return np.convolve(a_np, w_np)

def make_conv1d_fpga_func(M, N):
    return make_conv1d_fpga_function(make_conv1d_fpga_scheduler(M, N))


@pytest.mark.parametrize('execution_number', range(5))
def test1_M1_N1(execution_number):
    # Define dimension
    M = 1
    N = 1
    func = make_conv1d_fpga_func(M, N)

    # Create random test data
    np.random.seed(seed=execution_number)
    a_np = np.random.rand(M).astype(np.float32)
    w_np = np.random.rand(N).astype(np.float32)
    b_np = ans_np(a_np, w_np)
    b_out = func(a_np, w_np)

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


@pytest.mark.parametrize('execution_number', [1, 10, 100, 1000, 10000])
def test1_Mvar_N1024(execution_number):
    # Define dimension
    M = execution_number
    N = 1024
    func = make_conv1d_fpga_func(M, N)

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M).astype(np.float32)
    w_np = np.random.rand(N).astype(np.float32)
    b_np = ans_np(a_np, w_np)
    b_out = func(a_np, w_np)

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)


@pytest.mark.parametrize('execution_number', [1, 10, 100, 1000, 10000])
def test1_M1024_Nvar(execution_number):
    # Define dimension
    M = 1024
    N = execution_number
    func = make_conv1d_fpga_func(M, N)

    # Create random test data
    np.random.seed(seed=1024)
    a_np = np.random.rand(M).astype(np.float32)
    w_np = np.random.rand(N).astype(np.float32)
    b_np = ans_np(a_np, w_np)
    b_out = func(a_np, w_np)

    assert b_np.shape == b_out.shape, \
        "Shape mismatch: " + str(b_np.shape) + "\t" + str(b_out.shape)
    assert np.allclose(b_np, b_out), "Value mismatch: %s %s" % (b_np, b_out)
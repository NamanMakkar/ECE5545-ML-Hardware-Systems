import os
import sys
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

################################################################################
# Test for `count_trainable_parameters` function
################################################################################

from src.size_estimate import count_trainable_parameters
def test_count_trainable_parameters_linear_nobias():
    # Count Linear parameters
    net = nn.Linear(2, 3, bias=False)
    num_params = count_trainable_parameters(net)
    assert num_params == 6


def test_count_trainable_parameters_linear_bias():
    # Count Linear parameters
    net = nn.Linear(2, 3, bias=True)
    num_params = count_trainable_parameters(net)
    assert num_params == 9


def test_count_trainable_parameters_relu():
    # Count Linear parameters
    net = nn.ReLU()
    num_params = count_trainable_parameters(net)
    assert num_params == 0


def test_count_trainable_parameters_dropout():
    # Count Linear parameters
    net = nn.Dropout()
    num_params = count_trainable_parameters(net)
    assert num_params == 0


def test_count_trainable_parameters_conv2d_nobias():
    # Count Linear parameters
    net = nn.Conv2d(2, 3, 5, bias=False)
    num_params = count_trainable_parameters(net)
    assert num_params == 150


def test_count_trainable_parameters_conv2d_bias():
    # Count Linear parameters
    net = nn.Conv2d(2, 3, 5, bias=True)
    num_params = count_trainable_parameters(net)
    assert num_params == 153


def test_count_trainable_parameters_linear_bias_composed():
    # Count Linear parameters
    net = nn.Sequential(
        nn.Conv2d(2, 3, 5, bias=True),
        nn.Linear(3, 5, bias=False)
    )
    num_params = count_trainable_parameters(net)
    assert num_params == 168


################################################################################
# Test for `compute_forward_memory` function
################################################################################
from src.size_estimate import compute_forward_memory


def test_compute_forward_size_linear_nobias():
    net = nn.Linear(3, 5, bias=False)
    frd_memory = compute_forward_memory(net, (2, 3), 'cpu')
    # 4 bytes per float32; both input (2, 3) and output (2, 5) are in forward
    assert frd_memory == 64


def test_compute_forward_size_linear_bias():
    net = nn.Linear(3, 5, bias=True)
    frd_memory = compute_forward_memory(net, (2, 3), 'cpu')
    # 4 bytes per float32; both input (2, 3) and output (2, 5) are in forward
    assert frd_memory == 64


def test_compute_forward_size_relu():
    net = nn.ReLU()
    frd_memory = compute_forward_memory(net, (2, 3), 'cpu')
    # 4 bytes per float32; both input (2, 3) and output (2, 3) are in forward
    assert frd_memory == 48


def test_compute_forward_size_dropout():
    net = nn.Dropout()
    frd_memory = compute_forward_memory(net, (2, 3), 'cpu')
    # 4 bytes per float32; both input (2, 3) and output (2, 3) are in forward
    assert frd_memory == 48


def test_compute_forward_size_conv2d_nobiase():
    net = nn.Conv2d(3, 5, 3, padding=1, stride=1, bias=False)
    frd_memory = compute_forward_memory(net, (2, 3, 128, 128), 'cpu')
    # 4 bytes per float32; both input (2, 3) and output (2, 3) are in forward
    assert frd_memory == 1048576


def test_compute_forward_size_conv2d():
    net = nn.Conv2d(3, 5, 3, padding=1, stride=1, bias=True)
    frd_memory = compute_forward_memory(net, (2, 3, 128, 128), 'cpu')
    # 4 bytes per float32; both input (2, 3) and output (2, 3) are in forward
    assert frd_memory == 1048576


################################################################################
# Test for `flop` function
################################################################################
from src.size_estimate import flop


def test_flop_linear_nobias():
    net = nn.Linear(2, 3, bias=False)
    flop_by_layers = flop(net, input_shape=(10, 2), device='cpu')
    total_param_flops = sum([sum(val.values()) for val in flop_by_layers.values()])
    assert total_param_flops == 120


def test_flop_linear_bias():
    net = nn.Linear(2, 3, bias=True)
    flop_by_layers = flop(net, input_shape=(10, 2), device='cpu')
    total_param_flops = sum([sum(val.values()) for val in flop_by_layers.values()])
    assert total_param_flops == 150


def test_flop_conv_nobias():
    net = nn.Conv2d(2, 5, 3, padding=1, stride=1, bias=False)
    flop_by_layers = flop(net, input_shape=(10, 2, 4, 4), device='cpu')
    total_param_flops = sum([sum(val.values()) for val in flop_by_layers.values()])
    assert total_param_flops == 28800


def test_flop_conv_bias():
    net = nn.Conv2d(2, 5, 3, padding=1, stride=1, bias=True)
    flop_by_layers = flop(net, input_shape=(10, 2, 4, 4), device='cpu')
    total_param_flops = sum([sum(val.values()) for val in flop_by_layers.values()])
    assert total_param_flops == 29600

import os
import sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
EPS = 1e-8


################################################################################
# Test `ste_round` function backward pass
################################################################################
from src.quant import ste_round
ste_round_f = ste_round.apply


def test_ste_round_backward():
    x = torch.tensor(500)
    y = ste_round().backward(None, x)
    assert x.item() == y.item()
    assert id(x) != id(y)


################################################################################
# Test `linear_quantize` function
################################################################################
from src.quant import linear_quantize

def test_linear_quantize_1():
    zerop = torch.tensor([0])
    scale = torch.tensor([0.7])
    x = torch.tensor([1.2])
    y = linear_quantize(x, scale, zerop)
    ans = torch.tensor([2])
    assert x.shape == y.shape
    assert ((y - ans) ** 2).sum().item() < EPS


def test_linear_quantize_2():
    zerop = torch.tensor([10])
    scale = torch.tensor([0.7])
    x = torch.tensor([0.8])
    y = linear_quantize(x, scale, zerop)
    ans = torch.tensor([11])
    assert x.shape == y.shape
    assert ((y - ans) ** 2).sum().item() < EPS


def test_linear_quantize_3():
    zerop = torch.tensor([3])
    scale = torch.tensor([0.7])
    x = torch.tensor([-1.2])
    y = linear_quantize(x, scale, zerop)
    ans = torch.tensor([1])
    assert x.shape == y.shape
    assert ((y - ans) ** 2).sum().item() < EPS


################################################################################
# Test `SymmetricQuantFunction` Class
################################################################################
from src.quant import SymmetricQuantFunction
sym_quant = SymmetricQuantFunction.apply

def test_symmetric_quantization_1():
    zerop = torch.tensor([0])
    scale = torch.tensor([0.5])
    x = torch.tensor(
        [-0.2, 0.1, -0.7, 0.8, 2.2, -1.9],
        requires_grad=True
    )
    nbits = 2
    y = sym_quant(x, nbits, scale, zerop)
    assert y.shape == x.shape
    ans = torch.tensor([0, 0, -1, 1, 1, -2])
    assert ((y - ans) ** 2 < EPS).all().item()

    y.sum().backward()
    x_grad_ans = torch.tensor([2, 2, 2, 2, 2, 2])
    assert ((x.grad - x_grad_ans) ** 2 < EPS).all().item()


def test_symmetric_quantization_2():
    zerop = torch.tensor([0])
    scale = torch.tensor([0.5])
    x = torch.tensor([-1.2, 0.1, -0.7, 0.8, 2.2, -1.9], requires_grad=True)
    nbits = 3
    y = sym_quant(x, nbits, scale, zerop)
    assert y.shape == x.shape
    ans = torch.tensor([-2, 0, -1, 2, 3, -4])
    print(y, ans)
    assert ((y - ans) ** 2 < EPS).all().item()

    y.sum().backward()
    x_grad_ans = torch.tensor([2, 2, 2, 2, 2, 2])
    assert ((x.grad - x_grad_ans) ** 2 < EPS).all().item()


################################################################################
# Test `SymmetricQuantFunction` Class
################################################################################
from src.quant import AsymmetricQuantFunction
asym_quant = AsymmetricQuantFunction.apply


def test_asymmetric_quantization_1():
    zerop = torch.tensor([0])
    scale = torch.tensor([0.5])
    x = torch.tensor([-0.2, 0.1, -0.7, 0.8, 2.2, -1.9], requires_grad=True)
    nbits = 2
    y = asym_quant(x, nbits, scale, zerop)
    assert y.shape == x.shape
    ans = torch.tensor([0, 0, 0, 2, 3, 0])
    assert ((y - ans) ** 2 < EPS).all().item()

    y.sum().backward()
    x_grad_ans = torch.tensor([2, 2, 2, 2, 2, 2])
    assert ((x.grad - x_grad_ans) ** 2 < EPS).all().item()


def test_asymmetric_quantization_2():
    zerop = torch.tensor([-0.5])
    scale = torch.tensor([0.5])
    x = torch.tensor([-0.2, 0.1, -0.7, 0.8, 2.2, -1.9], requires_grad=True)
    nbits = 2
    y = asym_quant(x, nbits, scale, zerop)
    assert y.shape == x.shape
    ans = torch.tensor([0, 0, 0, 1, 3, 0])
    assert ((y - ans) ** 2 < EPS).all().item()

    y.sum().backward()
    x_grad_ans = torch.tensor([2, 2, 2, 2, 2, 2])
    assert ((x.grad - x_grad_ans) ** 2 < EPS).all().item()


def test_asymmetric_quantization_3():
    zerop = torch.tensor([1])
    scale = torch.tensor([0.5])
    x = torch.tensor([-0.2, 0.1, -0.7, 0.8, 2.2, -1.9], requires_grad=True)
    nbits = 2
    y = asym_quant(x, nbits, scale, zerop)
    assert y.shape == x.shape
    ans = torch.tensor([1, 1, 0, 3, 3, 0])
    print(y, ans)
    assert ((y - ans) ** 2 < EPS).all().item()

    y.sum().backward()
    x_grad_ans = torch.tensor([2, 2, 2, 2, 2, 2])
    assert ((x.grad - x_grad_ans) ** 2 < EPS).all().item()



################################################################################
# Test `QConfig` Class
################################################################################
from src.quant import QConfig


def test_qconfig_sym_1():
    qcfg = QConfig(quant_bits=4, is_symmetric=True)
    x = torch.tensor([-10., 12.])
    s, z = qcfg.get_quantization_params(x.min(), x.max())
    assert z.item() == 0 and abs(s.item() - 1.714285731) < EPS


def test_qconfig_sym_2():
    qcfg = QConfig(quant_bits=2, is_symmetric=True)
    x = torch.tensor([-10., 12.])
    s, z = qcfg.get_quantization_params(x.min(), x.max())
    assert z.item() == 0 and abs(s.item() - 12) < EPS


def test_qconfig_asym_1():
    qcfg = QConfig(quant_bits=4, is_symmetric=False)
    x = torch.tensor([-10., 12.])
    s, z = qcfg.get_quantization_params(x.min(), x.max())
    assert abs(z.item() - 7) < EPS and abs(s.item() - 1.466666698) < EPS


def test_qconfig_asym_2():
    qcfg = QConfig(quant_bits=3, is_symmetric=False)
    x = torch.tensor([-10., 12.])
    s, z = qcfg.get_quantization_params(x.min(), x.max())
    print(s.item(), z.item())
    assert abs(z.item() - 3) < EPS and abs(s.item() - 3.142857074) < EPS


################################################################################
# Test `quantize_weights_bias` function
################################################################################
from src.quant import quantize_weights_bias


def test_quantize_weights_bias_1():
    w = torch.tensor([-0.2, 0.5, 1.1, 0.2, -1, -2.2])
    qcfg = QConfig(quant_bits=2, is_symmetric=True)
    xqf = quantize_weights_bias(w, qcfg, fake_quantize=True)
    xqf_ans = torch.tensor([0., 0., 1.1, 0., -1.1, -2.2])
    assert ((xqf - xqf_ans) ** 2 < EPS).all().item()
    xq = quantize_weights_bias(w, qcfg, fake_quantize=False)
    xq_ans = torch.tensor([0, 0, 1, 0, -1, -2])
    assert ((xq - xq_ans) ** 2 < EPS).all().item()


def test_quantize_weights_bias_2():
    w = torch.tensor([-0.2, 0.5, 1.1, 0.2, -1, -2.2])
    qcfg = QConfig(quant_bits=2, is_symmetric=True)
    _ = quantize_weights_bias(w, qcfg, fake_quantize=True)
    w = torch.tensor([-0.2, 0.5, 1.1, 0.2, -1, -2.2, -3, 3])
    xqf = quantize_weights_bias(w, qcfg, fake_quantize=True)
    xq = quantize_weights_bias(w, qcfg, fake_quantize=False)

    xqf_ans = torch.tensor([0.,  0.,  0.,  0., 0., -3., -3.,  3.])
    assert ((xqf - xqf_ans) ** 2 < EPS).all().item()
    xq_ans = torch.tensor([0, 0, 0, 0, 0, -1, -1, 1])
    assert ((xq - xq_ans) ** 2 < EPS).all().item()


def test_quantize_weights_bias_3():
    w = torch.randn(10, 3, 5)
    qcfg = QConfig(quant_bits=2, is_symmetric=True)
    wqf = quantize_weights_bias(w, qcfg, fake_quantize=True)
    wq = quantize_weights_bias(w, qcfg, fake_quantize=False)
    assert wqf.shape == w.shape and wq.shape == w.shape


################################################################################
# Test `conv2d_linear_quantized` function
################################################################################
from src.quant import conv2d_linear_quantized


def test_conv2d_linear_quantized_linear():
    torch.manual_seed(0)
    layer = nn.Linear(3, 5)
    x = torch.tensor(
        [[-0.2, 0.5, 1.1], [0.2, -1, -2.2]],
        requires_grad=True)
    a_qcfg = QConfig(quant_bits=2, is_symmetric=False)
    w_qcfg = QConfig(quant_bits=2, is_symmetric=True)
    b_qcfg = QConfig(quant_bits=2, is_symmetric=True)
    y = conv2d_linear_quantized(
        layer, x, a_qconfig=a_qcfg, w_qconfig=w_qcfg, b_qconfig=b_qcfg)
    y_ans = torch.tensor([
        [-0.3357, 0.3464, 0.5143, -0.3464, -1.0178],
        [0.0000, 0.1786, 0.3464, -0.3464, -0.5143]
    ]).reshape(2, 5)
    assert (((y - y_ans) ** 2).mean() < EPS).all().item()

    y.sum().backward()
    x_grad_ans = torch.tensor([
        [-0.9155,  0.4578, -0.9155],
        [-0.9155,  0.4578, -0.9155]
    ])

    assert (((x.grad - x_grad_ans) ** 2).mean() < EPS).all().item()


def test_conv2d_linear_quantized_conv():
    torch.manual_seed(0)
    layer = nn.Conv2d(3, 5, kernel_size=1, padding=0, stride=1)
    x = torch.tensor(
        [[-0.2, 0.5, 1.1], [0.2, -1, -2.2]],
        requires_grad=True)
    a_qcfg = QConfig(quant_bits=2, is_symmetric=False)
    w_qcfg = QConfig(quant_bits=2, is_symmetric=True)
    b_qcfg = QConfig(quant_bits=2, is_symmetric=True)
    y = conv2d_linear_quantized(
        layer, x.reshape(2, 3, 1, 1),
        a_qconfig=a_qcfg, w_qconfig=w_qcfg, b_qconfig=b_qcfg)
    y = y.reshape(2, 5)
    y_ans = torch.tensor([
        [-0.3357, 0.3464, 0.5143, -0.3464, -1.0178],
        [0.0000, 0.1786, 0.3464, -0.3464, -0.5143]
    ]).reshape(2, 5)
    assert (((y - y_ans) ** 2).mean() < EPS).all().item()

    y.sum().backward()
    x_grad_ans = torch.tensor([
        [-0.9155,  0.4578, -0.9155],
        [-0.9155,  0.4578, -0.9155]
    ])

    assert (((x.grad - x_grad_ans) ** 2).mean() < EPS).all().item()

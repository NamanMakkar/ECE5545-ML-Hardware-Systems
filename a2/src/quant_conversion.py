import copy
import torch
import torch.nn as nn
from src.networks import Reshape
from src.quant import quantize_activations, quantize_weights_bias
from src.quant import QuantWrapper, QConfig


class Quant(nn.Module):
    """
    Module for quantization activations
    """

    def __init__(self, dtype=torch.float32, qconfig=None):
        super(Quant, self).__init__()
        self.qconfig = qconfig
        self.dtype = dtype

    def __repr__(self):
        s = super().__repr__()
        return f'{s}: qconfig={self.qconfig}; dtype={self.dtype}'

    def forward(self, x):
        if self.qconfig is None:
            return x
        out = quantize_activations(x, self.qconfig, is_moving_avg=False)
        return out.to(self.dtype)


class DeQuant(nn.Module):
    """
    Module that convert integers to floats.
    This is needed only because PyTorch Softmax does not support integers,
    so we need to convert integer values to float values before Softmax layer
    """

    def __init__(self, qconfig=None):
        super(DeQuant, self).__init__()
        self.qconfig = qconfig

    def forward(self, x):
        if self.qconfig is None:
            return x.to(torch.float32)
        else:
            raise NotImplementedError(
                f'Dequantization method not implemented for {self.qconfig}')


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def convert_to_int(qat_model, quant_bits, dtype=torch.int32):
    """
    Returns a new model with quantized nn.Conv2d and nn.Linear
    (all weights and bias are quantized), add Quant layers for activation
    quantization and DeQuant layer before Softmax
    """

    # Step 0: Freeze model and create an model for output.
    freeze_model(qat_model)
    new_model = nn.Sequential()

    # Step 1: add a quantization layer to quantize the input
    new_model.add_module('quant',
                         Quant(dtype, QConfig(quant_bits, is_symmetric=False)))

    # Step 2: iterate all layers and quantize them accordingly
    for i, (name, layer) in enumerate(qat_model.named_children()):
        if isinstance(layer, QuantWrapper):
            # Case 2.1: A Linear/Conv layer from quantized-aware finetuning
            new_model.add_module(f'{name}_quant', Quant(
                dtype, layer.a_qconfig))
            a_s = layer.a_qconfig.prev_scale

            # Step 2.1.2: Compute quantized parameters if scale and zero point
            # have already been calculated previously. If not, compute scale and
            # zero point. Once you've obtained the zero-point and scale, quantize
            # the weights and bias accordingly.
            module = copy.deepcopy(layer.module)
            w_q = layer.w_qconfig.quantize_with_prev_params(
                module.weight.data.detach(), fake_quantize=False)
            w_z = layer.w_qconfig.prev_zeropoint
            w_s = layer.w_qconfig.prev_scale

            b_q = layer.b_qconfig.quantize_with_prev_params(
                module.bias.data.detach(), fake_quantize=False)
            b_z = layer.b_qconfig.prev_zeropoint
            b_s = layer.b_qconfig.prev_scale
            bias = (b_q - b_z) * b_s / (w_s * a_s)

            # Step 2.1.3 Use the quantized weights and bias as parameters for
            # the Conv/Linear layer to be installed. Cast those quantized weights
            # to `dtype`. Then added the newly constructed layer to the new model.
            module.weight = nn.Parameter((w_q - w_z).to(dtype), False)
            module.bias = nn.Parameter(bias.to(dtype), False)
            module.type(dtype)
            new_model.add_module(name, module)

        elif isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # Case 2.2: A Linear/Conv layer from non-quantized-aware training
            # (e.g. fp32 finetuning). With this case, we directly

            # First add quantization layer for the input activation (convert it
            # to be integer)
            new_model.add_module(
                f'{name}_quant',
                Quant(dtype, QConfig(quant_bits, False)))

            layer = copy.deepcopy(layer)
            w_qconfig = QConfig(quant_bits, True)
            b_qconfig = QConfig(quant_bits, True)
            w_q = quantize_weights_bias(layer.weight.data.detach(), w_qconfig)
            b_q = quantize_weights_bias(layer.bias.data.detach(), b_qconfig)
            layer.weight = nn.Parameter(w_q.to(dtype), False)
            layer.bias = nn.Parameter(b_q.to(dtype), False)
            layer.type(dtype)
            new_model.add_module(name, layer)
        elif isinstance(layer, nn.Softmax):
            # Case 2.3: Since softmax can only take float number, we first add
            #           a DeQuant layer to convert integer to float before
            #           applying the softmax.
            new_model.add_module('dequant', DeQuant())
            new_model.add_module(name, layer)
        else:
            # Case 2.4 : ReLU/ReShape/Dropout will be kept the same.
            print("Staying the same:", name, layer)
            new_model.add_module(name, layer)
    return new_model


### Helper function for comparision

def compare_model(test_loader, model1, model2, device='cpu'):
    """
    Calculates the prediction accuracy for model1 and model2.
    Compute the percentage of same predictions they made
    """
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    num_same = 0
    num_correct1 = 0
    num_correct2 = 0
    for data, label in test_loader:
        data.to(device)
        label.to(device)
        predict1 = model1(data).argmax(dim=-1)
        predict2 = model2(data).argmax(dim=-1)
        num_same += predict1.squeeze().eq(predict2).sum().item()
        num_correct1 += predict1.squeeze().eq(label).sum().item()
        num_correct2 += predict2.squeeze().eq(label).sum().item()
    same_perc = num_same/len(test_loader.dataset)
    model1_acc = num_correct1/len(test_loader.dataset)
    model2_acc = num_correct2/len(test_loader.dataset)
    print(
        f'The models have {same_perc*100:.3f}% same predictions, \n' + \
        f'Model1 predicts {model1_acc*100:.3f}% of the samples correctly, \n' + \
        f'Model2 predicts {model2_acc*100:.3f}% of the samples correctly')
    return same_perc, model1_acc, model2_acc


def compare_model_mse(test_loader, model1, model2, device='cpu'):
    """
    Computes the MSE difference between model1's and model2's predictions
    """
    model1.to(device)
    model1.eval()
    model2.to(device)
    model2.eval()
    num_same = 0
    for data, label in test_loader:
        data.to(device)
        predict1 = model1(data)
        predict2 = model2(data)
        diff = nn.MSELoss()(predict1, predict2)
    print(f'MSE between two models\' prediction: {diff:.4f}')
    return diff


def print_features(sample_data, model, model_name='model', device='cpu'):
    # Print out min and max of activations and parameters
    model.eval()
    model.to(device)
    data = sample_data.to(device)
    print(f'features of {model_name} parameters:')
    for name, param in model.named_parameters():
        s = f'{name}: min={param.min()}; max={param.max()}; ' + \
                f'dtype={param.data.dtype}; shape={param.data.shape}'
        print(s)
    print(f'features of {model_name} activations/outputs')
    input_data = data
    print(f'input: min={input_data.data.min()}; max={input_data.data.max()}; ' + \
        f'dtype={input_data.data.dtype}; shape={input_data.data.shape}')
    for name, layer in model.named_children():
        input_data = layer(input_data)
        print(f'output of {name}: min={input_data.data.min()}; max={input_data.data.max()}; ' + \
            f'dtype={input_data.data.dtype}; shape={input_data.data.shape}')
import torch
import numpy as np
import torch.nn as nn


def flop(model, input_shape, device):
    total = {}

    def count_flops(name):
        def hook(module, input, output):
            "Hook that calculates number of floating point operations"
            flops = {}
            batch_size = input[0].shape[0]
            if isinstance(module, nn.Linear):
                # TODO: fill-in (start)
                input_feats = input[0].shape[1]
                output_feats = output.shape[1]
                if module.bias is not None:
                    flops[module] = batch_size * (2*input_feats * output_feats + output_feats)
                else:
                    flops[module] = batch_size * (2*input_feats * output_feats)

            if isinstance(module, nn.Conv2d):
                # TODO: fill-in (start)
                in_channels = input[0].shape[1]
                out_channels = output.shape[1]
                out_h = output.shape[2]
                out_w = output.shape[3]
                k_ops = 2 * module.kernel_size[0] * module.kernel_size[1] * (in_channels // module.groups)
                if module.bias is not None:
                    flops[module] = batch_size * (k_ops * out_channels * out_h * out_w + out_channels * out_h * out_w)
                else:
                    flops[module] = batch_size * k_ops * out_channels * out_h * out_w
                # TODO: fill-in (end)

            if isinstance(module, nn.BatchNorm1d):
                # TODO: fill-in (end)
                flops[module] = 4*input.numel()

            if isinstance(module, nn.BatchNorm2d):
                # TODO: fill-in (end)
                flops[module] = 4*input.numel()
            total[name] = flops
        return hook

    handle_list = []
    for name, module in model.named_modules():
        handle = module.register_forward_hook(count_flops(name))
        handle_list.append(handle)
    input = torch.ones(input_shape).to(device)
    model(input)

    # Remove forward hooks
    for handle in handle_list:
        handle.remove()
    return total


def count_trainable_parameters(model):
    """
    Return the total number of trainable parameters for [model]
    :param model:
    :return:
    """
    # TODO: fill-in (start)
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def compute_forward_memory(model, input_shape, device):
    """

    :param model:
    :param input_shape:
    :param device:
    :return:
    """
    
    # TODO: fill-in (start)
    input = torch.ones(input_shape)
    input_mem = np.prod(input_shape) * 4
    input = input.to(device)
    model = model.to(device)
    output_mem = np.prod(model(input).size()) * 4

    return input_mem + output_mem


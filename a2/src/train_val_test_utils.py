import os
import copy
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


def create_optimizer(model, learning_rate, optim_type='adam'):
    if optim_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    elif optim_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.7)
    else:
        raise ValueError("Invalid optim_type=%s" % optim_type)
    
    
def get_likely_index(tensor):
    # Find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def number_of_correct(pred, target):
    # Count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def plot_test_results(correct_list, total_list, audio_processor, title):
    total = sum(total_list)
    total_correct = sum(correct_list)

    correct_list = correct_list + [total_correct]
    total_list = total_list + [total]
    acc_array = np.array(correct_list) / np.array(total_list)
    acc_list = acc_array.tolist()

    perc_data = [total_list, correct_list, acc_list]

    col_names = audio_processor.words_list + ['total']
    rows = ['#samples', '#correct', 'accuracy']
    cell_text = []
    for row in range(len(rows)):
        cell_text.append(['{0:.3f}'.format(x) for x in perc_data[row]])
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.set_axis_off()
    table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=col_names,
                      rowColours = ["lightblue"]*3,
                      colColours = ["lightblue"]*5,
                      loc='upper left',
                      cellLoc='center')
    ax.set_title(title, fontweight = "bold")
    plt.show()


def plot_acc(test_loader, model, audio_processor, device, which_set='testing',
             model_name='model', model_type="float"):
    correct_list, total_list = test_model(
        model, test_loader, audio_processor, model_type=model_type, device=device)
    title = f'{which_set} accuracy for {model_name}'
    plot_test_results(correct_list, total_list, audio_processor, title)  
    return sum(correct_list) / sum(total_list)


def test_model(model, test_loader, audio_processor, model_type="float", device='cpu'):
    assert model_type in ["float", "half", "double"], f"model type {model_type} is not supported"
    model = copy.deepcopy(model)
    if model_type == "float":
        model.float()
    elif model_type == "double":
        model.double()
    elif model_type == "half":
        model.half()

    model.eval()
    model.to(device)

    num_labels = audio_processor.num_labels
    correct_list = [0 for i in range(num_labels)]
    total_list = [0 for i in range(num_labels)]

    for data, target in test_loader:

        if model_type == "float":
            data = data.float()
        elif model_type == "double":
            data = data.double()
        elif model_type == "half":
            data = data.half()

        data = data.to(device)
        target = target.to(device)

        # Apply model on whole batch directly on device
        output = model(data)

        pred = get_likely_index(output)

        for i in range(num_labels):
            target_mask = (i * torch.ones_like(target)) == target
            correct = (pred.squeeze().eq(target) * target_mask).sum().item()
            total_list[i] = total_list[i] + target_mask.sum().item()
            correct_list[i] = correct_list[i] + correct

    return correct_list, total_list


# Define train, valid, and test function
def train(model, loaders, optimizer, epoch, device,
          verbose=False):
    model.train()
    train_loader = loaders['training']
    valid_loader = loaders['validation']
    num_batches = len(train_loader)
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        # Calculate iteration_index
        iteration_index = (epoch - 1) * num_batches + batch_idx

        data = data.to(device)
        target = target.to(device)

        # Apply model on whole batch directly
        output = model(data)

        # Negative log-likelihood for a tensor of size (batch x 1 x n_output)
        target = target.to(dtype=torch.long)
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())
    
    acc = test(valid_loader, model, device, epoch=epoch,
               loader_type='Validation', verbose=verbose)
    print(f'Val Acc Epoch {epoch} = {round(acc,2)}%, Train loss = {round(sum(losses)/len(losses),3)}')

    return losses


def test(loader, model, device, epoch=None,
         loader_type='test', verbose=False):
    model.eval()
    correct = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        # Apply model on whole batch directly on device
        output = model(data)
        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

    # Update summary
    ttl = len(loader.dataset)
    acc = 100. * correct / ttl
    if verbose:
        print(f"\n{loader_type} epoch: {epoch}\tAccuracy: {correct}/{ttl} ({acc:.2f}%)\n")

    return acc


def choose_name(model_name):
    name = f"{model_name}_" + "{index}"
    i = 0
    while os.path.isfile(name.format(index=i)):
        i += 1
    name = name.format(index = i)
    print(f"The model's will be stored as: \n {name}")
    return name
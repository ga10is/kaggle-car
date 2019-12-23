import os
import math
import numpy as np
import pandas as pd
import torch

try:
    import matplotlib.pylab as plt
    import seaborn as sns
except ImportError:
    print('Unable import matplotlib and seaborn')


def save_checkpoint(state, is_best, dir_path):
    cp_path = os.path.join(dir_path, 'checkpoint.pth')
    torch.save(state, cp_path)
    if is_best:
        best_path = os.path.join(dir_path, 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(_model, _optimizer, _scheduler, fpath, _metric_fc=None):
    checkpoint = torch.load(fpath)
    # reset optimizer setting
    _epoch = checkpoint['epoch']
    _optimizer.load_state_dict(checkpoint['optimizer'])
    _scheduler.load_state_dict(checkpoint['scheduler'])
    _model.load_state_dict(checkpoint['state_dict'])
    if _metric_fc is not None:
        _metric_fc.load_state_dict(checkpoint['metric_fc'])

    return _epoch, _model, _optimizer, _scheduler, _metric_fc


'''Model Visualization'''


def normalize_channels(img):
    _min, _max = img.min(axis=(0, 1)), img.std(axis=(0, 1))
    img = (img - _min) / (_max - _min)
    return img


def plot_first_kernels(weight):
    ''' plot first filters of a model '''
    with torch.no_grad():
        filters = weight.detach().cpu().float().numpy(
        ).transpose([0, 2, 3, 1])  # channels last
        filters = normalize_channels(filters)
        filters /= filters.max()
    n = filters.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axs = plt.subplots(rows, cols)
    for c in range(cols):
        for r in range(rows):
            idx = r + c * rows
            if idx < n:
                axs[r, c].imshow(filters[idx])
            axs[r, c].set_axis_off()


def plot_norms(named_parameters, figsize=None):
    ''' plot l2 norm distribution for each layer of the given named parameters. e.g. model.named_parameters() '''
    from matplotlib import cm
    p = [0, 25, 50, 75, 100]
    with torch.no_grad():
        norms, names = [], []
        for name, param in named_parameters:
            param_flat = param.view(param.shape[0], -1)
            norms.append(np.percentile(torch.norm(
                param_flat, p=2, dim=1).cpu().numpy(), p))
            names.append(name)

    n = len(norms)
    inv_p = np.arange(len(p) - 1, -1, -1)
    norms = np.array(norms)
    if figsize is None:
        figsize = (np.min([16, n]), 6)
    plt.figure(figsize=figsize)
    plt.yscale('log')
    for i, c in zip(inv_p, cm.get_cmap('inferno')(0.1 + inv_p / len(p))):
        plt.bar(np.arange(n), norms[:, i], lw=1, color=c)
    plt.xticks(range(n), names, rotation="vertical")
    plt.xlabel("layers")
    plt.ylabel("norm distribution")
    plt.title("Kernel L2 Norms")
    plt.grid(True)
    plt.legend(labels=[f'{i}%' for i in p[::-1]])


def plot_grad_flow(named_parameters, figsize=None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/8#post_10'''
    from matplotlib.lines import Line2D
    avg_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.grad is not None) and ("bias" not in n):
            layers.append(n)
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    if figsize is None:
        figsize = (np.min([16, len(avg_grads)]), 6)
    plt.figure(figsize=figsize)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), avg_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(layers) + 1, lw=2, color="k")
    plt.xticks(range(0, len(layers), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(layers))
    plt.xlabel("Layers")
    plt.ylabel("Gradient Magnitude")
    plt.yscale('log')
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def cosine_distance_heatmap(model, x0, x1, y):
    ''' plot cosine distances between samples of 2 batches '''
    with torch.no_grad():
        typ = next(iter(model.parameters()))[0].type()
        f0 = model.features(x0.type(typ))
        f1 = model.features(x1.type(typ))
        cosX = 1 - torch.mm(f0, f1.t()).cpu().numpy()
        # del batch
        del f0
        del f1
        n = len(y)
        print('all-mean:', cosX.mean(), 'twins-mean:', cosX.trace() / n)

    idx = [f'{c}:{i:0>4}' for c, i in enumerate(y)]
    hm = pd.DataFrame(cosX, columns=idx, index=idx)
    fig = plt.figure(figsize=(n, n // 2))
    sns.heatmap(hm, annot=True, fmt=".2f")

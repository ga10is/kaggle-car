from tqdm import tqdm
import numpy as np
import torch

from . import config


def train_one_epoch(epoch, model, loader, criterion, optimizer):

    model.train()
    for i, data in enumerate(tqdm(loader)):
        data = data.to(device=config.DEVICE, dtype=torch.float)
        logit = model(data)
        loss = criterion(logit, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def valid_one_epoch(epoch, model, loader, criterion):
    loss_meter = AverageMeter()

    pred_dict = {}

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        data = data.to(config.DEVICE, dtype=torch.float)
        batch_size = data['heatmap'].size(0)
        with torch.no_grad():
            logit = model(data)
            loss = criterion(logit, data)
            loss_meter.update(loss.item(), batch_size)

            pred_batch_list = decode(logit)
            pred_batch_dict = dict(zip(img_ids, pred_batch_list))
            pred_dict.update(pred_batch_dict)

    # transform list of ndarray to ndarray
    for val in v6d_list:
        pred_dict[val] = np.concatenate(pred_dict[val])


def predict_one_epoch(model, loader):
    v6d_list = ['pitch', 'yaw', 'roll', 'x', 'y', 'z']

    pred_dict = {val: [] for val in v6d_list}

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        data = data.to(config.DEVICE, dtype=torch.float)
        batch_size = data['heatmap'].size(0)
        with torch.no_grad():
            logit = model(data)
            batch_dict = decode(logit)
            for val in v6d_list:
                pred_dict[val].append(batch_size[val].detach().cpu().numpy())
    # transform list of ndarray to ndarray
    for val in v6d_list:
        pred_dict[val] = np.concatenate(pred_dict[val])

    return pred_dict

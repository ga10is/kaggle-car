from tqdm import tqdm
import torch

from . import config
from .model.model import decode_train, decode_eval
from .model.metrics import car_map


def train_one_epoch(epoch, model, loader, criterion, optimizer):

    model.train()
    for i, data in enumerate(tqdm(loader)):
        img = data['image'].to(device=config.DEVICE)
        logit = model(img)
        loss, loss_stats = criterion(logit, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def valid_one_epoch(epoch, model, loader, criterion):
    loss_meter = AverageMeter()

    pred_dict = {}
    gt_dict = {}

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img = data['image'].to(config.DEVICE)
        batch_size = data['image'].size(0)
        with torch.no_grad():
            logit = model(img)
            loss = criterion(logit, data)
            loss_meter.update(loss.item(), batch_size)

            pred_batch_list = decode_eval(logit)
            img_ids = data['ImageId']
            pred_batch_dict = dict(zip(img_ids, pred_batch_list))
            pred_dict.update(pred_batch_dict)
            gt_batch_dict = dict(zip(img_ids, data['gt']))
            gt_dict.update(gt_batch_dict)
    map_val = car_map(gt_dict, pred_dict)
    return map_val


def predict_one_epoch(model, loader):
    pred_dict = {}

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img = data['image'].to(config.DEVICE)
        with torch.no_grad():
            logit = model(img)

            pred_batch_list = decode_eval(logit)
            img_ids = data['ImageId']
            pred_batch_dict = dict(zip(img_ids, pred_batch_list))
            pred_dict.update(pred_batch_dict)

    return pred_dict

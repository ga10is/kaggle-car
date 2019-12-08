from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .data.dataset import CarDataset, car_collate_fn, get_train_transform
from .model.model import decode_eval, ResNet
from .model.metrics import car_map, AverageMeter
from .model.loss import CarLoss


def train():
    df_train = pd.read_csv(config.TRAIN_CSV)

    train_dataset = CarDataset(
        df_train, config.TRAIN_IMAGE, 'train', get_train_transform())
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              collate_fn=car_collate_fn
                              )

    model, criterion, optimizer, scheduler = init_model()

    start_epoch = 0
    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer)


def init_model():
    model = ResNet().to(config.DEVICE)
    criterion = CarLoss()
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}], lr=config.ADAM_LR)
    scheduler = None

    return model, criterion, optimizer, scheduler


def train_one_epoch(epoch, model, loader, criterion, optimizer):
    loss_meters = {k: AverageMeter() for k in []}

    model.train()
    for i, data in enumerate(tqdm(loader)):
        batch_size = data['image'].size(0)
        to_gpu(data)

        logit = model(data['image'])
        loss, loss_stats = criterion(logit, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in loss_stats:
            loss_meters[k].update(loss_stats[k].item(), batch_size)

        print_loss(i, loss_meters)


def to_gpu(data):
    for k in ['image', 'heatmap', 'offset', 'xyz', 'rotate', 'index', 'rot_mask', 'reg_mask']:
        data[k] = data[k].to(config.DEVICE)

# def update_meter()


def print_loss(idx, loss_meters):
    if (idx + 1) % config.PRINT_FREQ == 0:
        print('loss %f heatmap %f offset %f depth %f'
              % (loss_meters['loss'].avg,
                 loss_meters['loss_heatmap'].avg,
                 loss_meters['loss_offset'].avg,
                 loss_meters['loss_depth'].avg))


def valid_one_epoch(epoch, model, loader, criterion):
    loss_meter = AverageMeter()

    pred_dict = {}
    gt_dict = {}

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        to_gpu(data)
        batch_size = data['image'].size(0)
        with torch.no_grad():
            logit = model(data['image'])
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

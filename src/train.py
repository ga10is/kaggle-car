import pprint
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from apex import amp

from . import config
from .common.util import str_stats
from .data.dataset import CarDataset, train_collate_fn, get_train_transform
from .model.model import decode_eval, _nms
from .model.model_util import save_checkpoint, load_checkpoint
from .model.metrics import car_map, AverageMeter
from .model.loss import CarLoss, _sigmoid
from .model.hourglass import get_large_hourglass_net


def train():
    # ID_1a5a10365.jpg, ID_4d238ae90.jpg, ID_408f58e9f.jpg, ID_bb1d991f6.jpg, ID_c44983aeb.jpg
    df_train = pd.read_csv(config.TRAIN_CSV)
    broken_images = ['ID_1a5a10365', 'ID_4d238ae90',
                     'ID_408f58e9f', 'ID_bb1d991f6', 'ID_c44983aeb']
    df_train = df_train[~df_train['ImageId'].isin(broken_images)]
    df_trn, df_val = train_test_split(df_train, test_size=0.1)

    train_dataset = CarDataset(
        df_trn, config.TRAIN_IMAGE, 'train', get_train_transform())
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=train_collate_fn,
                              )
    valid_dataset = CarDataset(
        df_val, config.TRAIN_IMAGE, 'valid', get_train_transform())
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=train_collate_fn
                              )
    start_epoch = 0
    best_score = 0

    model, criterion, optimizer, scheduler = initialize()

    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer)

        valid_score = valid_one_epoch(epoch, model, valid_loader, criterion)

        # check point
        is_best = valid_score > best_score
        if is_best:
            best_score = valid_score
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, config.OUTDIR_PATH)

        scheduler.step()


def initialize():
    model, criterion, optimizer, scheduler = init_model()
    if config.USE_PRETRAINED:
        start_epoch, model, optimizer, scheduler, _ = load_checkpoint(
            model, optimizer, scheduler, config.PRETRAIN_PATH)
        print('loaded: %s(epoch: %d)' % (config.PRETRAIN_PATH, start_epoch))

        if config.RESET_OPT:
            start_epoch, optimizer, scheduler = reset_opt(model)

    return model, criterion, optimizer, scheduler


def init_model():
    model = get_large_hourglass_net().to(config.DEVICE)
    criterion = CarLoss()
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}], lr=config.ADAM_LR)
    # scheduler = None
    mile_stones = [5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, mile_stones, gamma=0.1, last_epoch=-1)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    return model, criterion, optimizer, scheduler


def reset_opt(model):
    print('reset optimizer')
    start_epoch = 0
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}], lr=config.ADAM_LR)
    mile_stones = [1, 5]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, mile_stones, gamma=0.1, last_epoch=-1)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    return start_epoch, optimizer, scheduler


def train_one_epoch(epoch, model, loader, criterion, optimizer):
    loss_meters = {k: AverageMeter()
                   for k in ['loss', 'loss_heatmap', 'loss_heatmap_reg', 'loss_offset', 'loss_depth', 'loss_rotate']}
    lr = optimizer.state_dict()['param_groups'][0]['lr']

    print('[Start] epoch: %d' % epoch)
    print('lr: %f' % lr)

    model.train()
    for i, data in enumerate(tqdm(loader)):
        batch_size = data['image'].size(0)
        to_gpu(data)
        with torch.set_grad_enabled(True):
            logits = model(data['image'])
            loss, loss_stats = criterion(logits, data)

            optimizer.zero_grad()
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

        for k in loss_stats:
            loss_meters[k].update(loss_stats[k].item(), batch_size)

        if (i + 1) % config.PRINT_FREQ == 0:
            print_loss(i, loss_meters)
            # last output of Hourglass is used
            show_heatmap(data, logits[-1])
            print_decode(data, logits[-1])


def to_gpu(data):
    for k in ['image', 'heatmap', 'offset', 'depth', 'rotate', 'index', 'rot_mask', 'reg_mask']:
        data[k] = data[k].to(config.DEVICE)


def print_loss(idx, loss_meters):
    print('loss %f heatmap %f heatmap(l1) %f offset %f depth %f rotate %f'
          % (loss_meters['loss'].avg,
             loss_meters['loss_heatmap'].avg,
             loss_meters['loss_heatmap_reg'].avg,
             loss_meters['loss_offset'].avg,
             loss_meters['loss_depth'].avg,
             loss_meters['loss_rotate'].avg)
          )


def print_decode(data, output):
    # gt = data['gt'][0]
    depth = data['depth'][0].detach().cpu().numpy()
    decode_output = decode_eval(output, k=2, on_nms=True)[0]

    pprint.pprint(decode_output)
    print(str_stats(depth.reshape(-1)))


def show_heatmap(data, output):
    img = data['image'][0].detach().cpu().numpy()
    # transpose (c, h, w) -> (h, w, c)
    img = img.transpose(1, 2, 0)
    # unnormalize
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    img = img * std + mean

    heatmaps = output['heatmap'].detach().cpu()
    print('logit heatmap', str_stats(heatmaps.view(-1).numpy()))
    heatmaps = _sigmoid(heatmaps)
    print('sigmoid heatmap', str_stats(heatmaps.view(-1).numpy()))
    heatmap = heatmaps[0].numpy()
    # transform (c, h, w) -> (h, w)
    heatmap = heatmap[0, :, :]

    # min-max normalization
    # hm_max = heatmap.max()
    # hm_min = heatmap.min()
    # heatmap = (heatmap - hm_min) / (hm_max - hm_min)
    # heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # depth
    depth = output['depth'][0].detach().cpu()
    depth = 1. / (depth.sigmoid() + 1e-6) - 1
    depth = depth.numpy()[0, :, :]

    gt_heatmap = data['heatmap'][0].detach()
    gt_heatmap = gt_heatmap.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes[0, 0].imshow(img)
    axes[0, 1].imshow(gt_heatmap)
    axes[1, 0].imshow(heatmap)
    axes[1, 1].imshow(depth)
    plt.show()


def valid_one_epoch(epoch, model, loader, criterion):
    loss_meter = AverageMeter()

    pred_dict = {}
    nms_pred_dict = {}
    gt_dict = {}

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        to_gpu(data)
        with torch.no_grad():
            logits = model(data['image'])
            # loss = criterion(logit, data)
            # loss_meter.update(loss.item(), batch_size)

        # last output of Hourglass is used
        pred_batch_list = decode_eval(
            logits[-1], k=config.MAX_OBJ, on_nms=False)
        nms_pred_batch_list = decode_eval(
            logits[-1], k=config.MAX_OBJ, on_nms=True)

        img_ids = data['ImageId']
        pred_batch_dict = dict(zip(img_ids, pred_batch_list))
        pred_dict.update(pred_batch_dict)

        nms_pred_batch_dict = dict(zip(img_ids, nms_pred_batch_list))
        nms_pred_dict.update(nms_pred_batch_dict)

        gt_batch_dict = dict(zip(img_ids, data['gt']))
        gt_dict.update(gt_batch_dict)

    map_val = car_map(gt_dict, pred_dict)
    print('mAP: %f' % (map_val, ))

    nms_map_val = car_map(gt_dict, nms_pred_dict)
    print('mAP(nms): %f' % (nms_map_val, ))

    return map_val

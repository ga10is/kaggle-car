import pprint
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from apex import amp

from . import config
from .data.dataset import CarDataset, car_collate_fn, get_train_transform
from .model.model import decode_eval, ResNet, _nms
from .model.metrics import car_map, AverageMeter
from .model.loss import CarLoss, _sigmoid


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
                              collate_fn=car_collate_fn,
                              )
    valid_dataset = CarDataset(
        df_val, config.TRAIN_IMAGE, 'valid', get_train_transform())
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=car_collate_fn
                              )

    model, criterion, optimizer, scheduler = init_model()

    start_epoch = 0
    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        train_one_epoch(epoch, model, train_loader, criterion, optimizer)

        valid_one_epoch(epoch, model, valid_loader, criterion)


def init_model():
    model = ResNet().to(config.DEVICE)
    criterion = CarLoss()
    optimizer = torch.optim.Adam(
        [{'params': model.parameters()}], lr=config.ADAM_LR)
    scheduler = None
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    return model, criterion, optimizer, scheduler


def train_one_epoch(epoch, model, loader, criterion, optimizer):
    loss_meters = {k: AverageMeter()
                   for k in ['loss', 'loss_heatmap', 'loss_offset', 'loss_depth', 'loss_rotate']}

    model.train()
    for i, data in enumerate(tqdm(loader)):
        batch_size = data['image'].size(0)
        to_gpu(data)
        with torch.set_grad_enabled(True):
            logits = model(data['image'])
            loss, loss_stats = criterion(logits, data)

            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            # scaled_loss.backward()
            if (i + 1) % config.ACC_ITER == 0:
                optimizer.step()
                optimizer.zero_grad()

        for k in loss_stats:
            loss_meters[k].update(loss_stats[k].item(), batch_size)

        print_loss(i, loss_meters)
        if (i + 1) % 10 == 0:
            # last output of Hourglass is used
            show_heatmap(data, logits[-1])
            print_decode(data, logits[-1])


def to_gpu(data):
    for k in ['image', 'heatmap', 'offset', 'depth', 'rotate', 'index', 'rot_mask', 'reg_mask']:
        data[k] = data[k].to(config.DEVICE)

# def update_meter()


def print_loss(idx, loss_meters):
    if (idx + 1) % config.PRINT_FREQ == 0:
        print('loss %f heatmap %f offset %f depth %f rotate %f'
              % (loss_meters['loss'].avg,
                 loss_meters['loss_heatmap'].avg,
                 loss_meters['loss_offset'].avg,
                 loss_meters['loss_depth'].avg),
              loss_meters['loss_rotate'].avg)


def print_decode(data, output):
    gt = data['gt'][0]
    depth = data['depth'][0].detach().cpu().numpy()
    decode_output = decode_eval(output, k=10)[0]
    pprint.pprint(gt)
    print('depth')
    print(depth)
    pprint.pprint(decode_output)


def show_heatmap(data, output):
    img = data['image'][0].detach().cpu().numpy()
    # transpose (c, h, w) -> (h, w, c)
    img = img.transpose(1, 2, 0)
    # unnormalize
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    img = img * std + mean

    heatmaps = output['heatmap'].detach()
    heatmap = _sigmoid(heatmaps[0]).cpu().numpy()
    nms_heatmap = _nms(heatmaps)[0].cpu().numpy()
    # transform (c, h, w) -> (h, w)
    heatmap = heatmap[0, :, :]
    nms_heatmap = nms_heatmap[0, :, :]

    heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    gt_heatmap = data['heatmap'][0].detach()
    gt_heatmap = gt_heatmap.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes[0, 0].imshow(img)
    axes[0, 1].imshow(gt_heatmap)
    axes[1, 0].imshow(heatmap)
    axes[1, 1].imshow(nms_heatmap)
    plt.show()


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
            logits = model(data['image'])
            # loss = criterion(logits, data)
            # loss_meter.update(loss.item(), batch_size)

        # last output of Hourglass is used
        pred_batch_list = decode_eval(logits[-1], k=config.MAX_OBJ)
        img_ids = data['ImageId']
        pred_batch_dict = dict(zip(img_ids, pred_batch_list))
        pred_dict.update(pred_batch_dict)
        gt_batch_dict = dict(zip(img_ids, data['gt']))
        gt_dict.update(gt_batch_dict)
    map_val = car_map(gt_dict, pred_dict)

    print('mAP: %f' % (map_val, ))
    return map_val

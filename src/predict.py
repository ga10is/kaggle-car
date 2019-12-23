from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .model.model import decode_eval
from .model.model_util import load_checkpoint
from .data.dataset import CarDataset, car_collate_fn, get_train_transform
from .train import init_model


def predict():
    df_test = pd.read_csv(config.SUBMIT_CSV)
    test_dataset = CarDataset(
        df_test, config.TEST_IMAGE, 'predict', get_train_transform())
    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE,
                             shuffle=False,
                             num_workers=config.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=car_collate_fn
                             )

    model, criterion, optimizer, scheduler = init_model()
    load_checkpoint(model, optimizer, scheduler, config.PRETRAIN_PATH)

    pred_dict = predict_one_epoch(model, test_loader)


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


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

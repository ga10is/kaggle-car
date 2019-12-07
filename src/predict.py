from tqdm import tqdm
import torch
# from torch.utils.data import DataLoader

from . import config
from .model.model import decode_eval
# from .data.dataset import CarDataset, car_collate_fn


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

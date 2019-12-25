from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import config
from .model.model import decode_eval
from .model.model_util import load_checkpoint
from .data.dataset import CarDataset, test_collate_fn, get_test_transform
from .train import init_model


def predict():
    df_test = pd.read_csv(config.SUBMIT_CSV)
    test_dataset = CarDataset(
        df_test, config.TEST_IMAGE, 'test', get_test_transform())
    test_loader = DataLoader(test_dataset,
                             batch_size=config.BATCH_SIZE,
                             shuffle=False,
                             num_workers=config.NUM_WORKERS,
                             pin_memory=True,
                             drop_last=False,
                             collate_fn=test_collate_fn
                             )

    model, criterion, optimizer, scheduler = init_model()
    load_checkpoint(model, optimizer, scheduler, config.PRETRAIN_PATH)
    print('loaded: %s' % config.PRETRAIN_PATH)

    pred_dict = predict_one_epoch(model, test_loader)
    output_csv(pred_dict)


def predict_one_epoch(model, loader):
    pred_dict = {}

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img = data['image'].to(config.DEVICE)
        with torch.no_grad():
            outputs = model(img)

            pred_batch_list = decode_eval(
                outputs[-1], k=config.MAX_OBJ, on_nms=True)
            img_ids = data['ImageId']
            pred_batch_dict = dict(zip(img_ids, pred_batch_list))
            pred_dict.update(pred_batch_dict)

    return pred_dict


def output_csv(pred_dict):
    pred_list = [(k, coords2str(v)) for k, v in pred_dict.items()]
    df_pred = pd.DataFrame(pred_list, columns=['ImageId', 'PredictionString'])
    print(df_pred.head())
    df_pred.to_csv('submission.csv', index=False)


def coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence']):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from math import acos, pi
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
from multiprocessing import Pool


def expand_df(df, PredictionStringCols):
    df = df.dropna().copy()
    df['NumCars'] = [int((x.count(' ') + 1) / 7)
                     for x in df['PredictionString']]

    image_id_expanded = [item for item, count in zip(
        df['ImageId'], df['NumCars']) for i in range(count)]
    prediction_strings_expanded = df['PredictionString'].str.split(
        ' ', expand=True).values.reshape(-1, 7).astype(float)
    # get records which don't have at least one nan in the record
    prediction_strings_expanded = prediction_strings_expanded[
        ~np.isnan(prediction_strings_expanded).all(axis=1)]
    df = pd.DataFrame(
        {
            'ImageId': image_id_expanded,
            PredictionStringCols[0]: prediction_strings_expanded[:, 0],
            PredictionStringCols[1]: prediction_strings_expanded[:, 1],
            PredictionStringCols[2]: prediction_strings_expanded[:, 2],
            PredictionStringCols[3]: prediction_strings_expanded[:, 3],
            PredictionStringCols[4]: prediction_strings_expanded[:, 4],
            PredictionStringCols[5]: prediction_strings_expanded[:, 5],
            PredictionStringCols[6]: prediction_strings_expanded[:, 6]
        })
    return df


def str2coords(s, names):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
    return coords


def TranslationDistance(p, g, abs_dist=False):
    dx = p['x'] - g['x']
    dy = p['y'] - g['y']
    dz = p['z'] - g['z']
    diff0 = (g['x']**2 + g['y']**2 + g['z']**2)**0.5
    diff1 = (dx**2 + dy**2 + dz**2)**0.5
    if abs_dist:
        diff = diff1
    else:
        diff = diff1 / diff0
    return diff


def RotationDistance(p, g):
    true = [g['pitch'], g['yaw'], g['roll']]
    pred = [p['pitch'], p['yaw'], p['roll']]
    q1 = R.from_euler('xyz', true)
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)

    # in the official metrics code:
    # https://www.kaggle.com/c/pku-autonomous-driving/overview/evaluation
    #   return Object3D.RadianToDegree( Math.Acos(diff.W) )
    # this code treat θ and θ+2π differntly.
    # So this should be fixed as follows.
    W = (acos(W) * 360) / pi
    if W > 180:
        W = 360 - W
    return W


thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]


def check_match(idx):
    keep_gt = False
    thre_tr_dist = thres_tr_list[idx]
    thre_ro_dist = thres_ro_list[idx]
    train_dict = {imgID: str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])
                  for imgID, s in zip(train_df['ImageId'], train_df['PredictionString'])}
    valid_dict = {imgID: str2coords(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score'])
                  for imgID, s in zip(valid_df['ImageId'], valid_df['PredictionString'])}
    result_flg = []  # 1 for TP, 0 for FP
    scores = []
    MAX_VAL = 10**10
    for img_id in valid_dict:
        for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
            # find nearest GT
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate(train_dict[img_id]):
                tr_dist = TranslationDistance(pcar, gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    # It is better to calculate rotate distance
                    # when idx according to the minimum of dist is determined.
                    min_ro_dist = RotationDistance(pcar, gcar)
                    min_idx = idx

            # set the result
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    # if
                    train_dict[img_id].pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            scores.append(pcar['carid_or_score'])

    return result_flg, scores


def car_map(valid_predict):
    valid_df = pd.read_csv(valid_predict)
    valid_df = valid_df.fillna('')

    train_df = pd.read_csv('../input/pku-autonomous-driving/train.csv')
    train_df = train_df[train_df.ImageId.isin(valid_df.ImageId.unique())]
    # data description page says, The pose information is formatted as
    # model type, yaw, pitch, roll, x, y, z
    # but it doesn't, and it should be
    # model type, pitch, yaw, roll, x, y, z
    expanded_train_df = expand_df(
        train_df, ['model_type', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])

    max_workers = 10
    n_gt = len(expanded_train_df)
    ap_list = []
    p = Pool(processes=max_workers)
    for result_flg, scores in p.imap(check_match, range(10)):
        if np.sum(result_flg) > 0:
            n_tp = np.sum(result_flg)
            recall = n_tp / n_gt
            ap = average_precision_score(result_flg, scores) * recall
        else:
            ap = 0
        ap_list.append(ap)
    map = np.mean(ap_list)
    print('map:', map)

    return map


def check_match2(idx, gt_dict_org, pred_dict_org):
    keep_gt = False
    thre_tr_dist = thres_tr_list[idx]
    thre_ro_dist = thres_ro_list[idx]
    # copy() because gt_dict is called pop()
    gt_dict = gt_dict_org.copy()
    pred_dict = pred_dict_org.copy()

    result_flg = []  # 1 for TP, 0 for FP
    scores = []
    MAX_VAL = 10**10
    for img_id in pred_dict:
        for pcar in pred_dict[img_id]:
            # find nearest GT
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate(gt_dict[img_id]):
                tr_dist = TranslationDistance(pcar, gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    min_idx = idx
            # Calculate rotate distance of the nearest object
            min_gcar = gt_dict[img_id][min_idx]
            min_ro_dist = RotationDistance(pcar, min_gcar)

            # set the result
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    # if
                    gt_dict[img_id].pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            scores.append(pcar['carid_or_score'])

    return result_flg, scores


def check_match_wrapper(params):
    check_match2(*params)


def car_map2(valid_predict):
    valid_df = pd.read_csv(valid_predict)
    valid_df = valid_df.fillna('')

    train_df = pd.read_csv('../input/pku-autonomous-driving/train.csv')
    train_df = train_df[train_df.ImageId.isin(valid_df.ImageId.unique())]
    # data description page says, The pose information is formatted as
    # model type, yaw, pitch, roll, x, y, z
    # but it doesn't, and it should be
    # model type, pitch, yaw, roll, x, y, z
    expanded_train_df = expand_df(
        train_df, ['model_type', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])

    gt_dict = {imgID: str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z'])
               for imgID, s in zip(train_df['ImageId'], train_df['PredictionString'])}
    pred_dict = {imgID: str2coords(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score'])
                 for imgID, s in zip(valid_df['ImageId'], valid_df['PredictionString'])}
    # sort values of pred_dict for each image
    for img_id in pred_dict:
        pred_dict[img_id] = sorted(
            pred_dict[img_id], key=lambda x: -x['carid_or_score'])

    arguments = [(i, gt_dict, pred_dict) for i in range(10)]
    max_workers = 10
    n_gt = len(expanded_train_df)
    ap_list = []
    with Pool(processes=max_workers) as p:
        for result_flg, scores in p.imap(check_match_wrapper, arguments):
            if np.sum(result_flg) > 0:
                n_tp = np.sum(result_flg)
                recall = n_tp / n_gt
                ap = average_precision_score(result_flg, scores) * recall
            else:
                ap = 0
            ap_list.append(ap)
        map = np.mean(ap_list)
        print('map:', map)

    return map

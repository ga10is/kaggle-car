import numpy as np


def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords


CAMERA = np.array([[2304.5479, 0, 1686.2379],
                   [0, 2305.8757, 1354.9849],
                   [0, 0, 1]], dtype=np.float32)


def project(coord_list):
    xs = [c['x'] for c in coord_list]
    ys = [c['y'] for c in coord_list]
    zs = [c['z'] for c in coord_list]
    P = np.array(list(zip(xs, ys, zs))).T
    # project
    img_p = np.dot(CAMERA, P).T
    # normalize z -> 1
    # x' = x / z, y' = y / z
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2]
    return img_xs, img_ys, img_zs

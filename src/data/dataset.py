import os

# import torch
from torch.utils.data import Dataset
# from albumentations import ImageOnlyTransform
# import albumentations.pytorch as ATorch
# import albumentations as A
import numpy as np
import cv2
# import jpeg4py

from .. import config
from .cmp_util import str2coords, project


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m: m + 1, -n: n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    mask_range = (y - top, y + bottom, x - left, x + right)
    masked_heatmap = heatmap[y - top: y + bottom, x - left:x + right]
    masked_gaussian = gaussian[
        radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(
            heatmap[mask_range[0]: mask_range[1],
                    mask_range[2]: mask_range[3]],
            masked_gaussian * k,
            out=heatmap[mask_range[0]: mask_range[1],
                        mask_range[2]: mask_range[3]]
        )
    return heatmap


class CarDataset(Dataset):
    def __init__(self, df, image_dir, transform, mode):
        self.df_org = df.copy()
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode

        # Random Selection
        if mode == 'train':
            # self.update()
            self.df_selected = self.df_org
        elif mode in ['valid', 'predict']:
            self.df_selected = self.df_org
        else:
            raise ValueError('Unexpected mode: %s' % mode)

    def __len__(self):
        return self.df_selected.shape[0]

    def __getitem__(self, idx):
        image_path = self._get_image_name(self.df_selected, idx)
        try:
            image = self._load_image(image_path)
        except Exception as e:
            raise ValueError('Could not load image: %s' % image_path) from e

        img_height, img_width = image.shape[0], image.shape[1]
        hm_height, hm_width = img_height // config.MODEL_SCALE, img_width // config.MODEL_SCALE

        # index: 1-dim index of the heatmap from top-left to right-bottom
        heatmap = np.zeros(
            (hm_height, hm_width), dtype=np.float32)
        offset = np.zeros((config.MAX_OBJ, 2), dtype=np.float32)
        xyz = np.zeros((config.MAX_OBJ, 3), dtype=np.float32)
        rotate = np.zeros((config.MAX_OBJ, 4), dtype=np.float32)
        index = np.zeros((config.MAX_OBJ), dtype=np.uint8)
        rot_mask = np.zeros((config.MAX_OBJ), dtype=np.uint8)

        # set groud-truth data
        coord_str = self.df_selected.iloc[idx]['PredictionString']
        coord_list = str2coords(coord_str)
        img_xs, img_ys, img_zs = project(coord_list)

        # remove point in out of image range
        valid_xs, valid_ys, valid_zs = [], [], []
        for img_x, img_y, img_z in zip(img_xs, img_ys, img_zs):
            if img_x > 0 and img_y > 0:
                valid_xs.append(img_x)
                valid_ys.append(img_y)
                valid_zs.append(img_z)

        num_objs = min(len(valid_xs), config.MAX_OBJ)
        for k in range(num_objs):
            center = np.array(
                [valid_xs[k] / config.MODEL_SCALE, valid_ys[k] / config.MODEL_SCALE], dtype=np.float32)
            center_int = center.astype(np.int32)
            # z value 0 ~ 3500
            radius = int(2000 / valid_zs[k] / config.MODEL_SCALE)
            # radius = max(1, radius)

            heatmap = draw_umich_gaussian(heatmap, center, radius)
            offset[k] = center - center_int
            xyz[k] = np.array([valid_xs[k], valid_ys[k], valid_zs[k]])
            index[k] = center_int[1] * config.OUTPUT_WIDTH + center_int[0]
            rot_mask[k] = 1

        ret = {
            'image': image,
            'heatmap': heatmap,
            'offset': offset,
            'xyz': xyz,
            'rotate': rotate,
            'index': index,
            'rot_mask': rot_mask,
            'ImageId': self._get_image_id(self.df_selected, idx)
        }

        return ret

    def _get_image_name(self, df, idx):
        """
        get image name

        Returns
        -------
        file_path: str
            image file path
        """
        rcd = df.iloc[idx]
        image_id = rcd['ImageId']

        file_name = '%s.jpg' % image_id
        file_path = os.path.join(self.image_dir, file_name)
        return file_path

    def _load_image(self, image_path):
        """
        Parameters
        ----------
        image_path: str
            image file path
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = jpeg4py.JPEG(image_path).decode()

        if image is None:
            raise ValueError('Not found image: %s' % image_path)

#         augmented = self.transform(image=image)
#         image = augmented['image']
        return image

    def _get_image_id(self, df, idx):
        rcd = df.iloc[idx]
        image_id = rcd['ImageId']
        return image_id

    def plot_2d(self, idx):
        image_name = self._get_image_name(self.df_selected, idx)
        try:
            image = self._load_image(image_name)
        except Exception as e:
            raise ValueError('Could not load image: %s' % image_name) from e

        coord_str = self.df_selected.iloc[idx]['PredictionString']
        coord_list = str2coords(coord_str)
        img_xs, img_ys = project(coord_list)

        plt.figure(figsize=(14, 14))
        plt.imshow(image)
        plt.scatter(x=img_xs, y=img_ys, color='red', s=100)

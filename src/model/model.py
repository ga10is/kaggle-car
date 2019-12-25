import math
import torch.nn as nn
import torchvision
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

from .. import config
# from .dcn.dcn_v2 import DCN
from .loss import _gather_feat, _transpose_and_gather_feat, _sigmoid
from ..data.cmp_util import CAMERA

BN_MOMENTUM = 0.1


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 512
        self.heads = {'heatmap': 1, 'offset': 2, 'depth': 1, 'rotate': 4}
        self.deconv_with_bias = False

        resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, classes, kernel_size=1,
                          stride=1, padding=0, bias=True)
            )
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes,
                     kernel_size=(3, 3), stride=1,
                     padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1,
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_eval(output, k, on_nms):
    """
    Parameters
    ----------
    output: dict
        output contains the following key-values.
        heatmap: torch.Tensor, size of (batch_size, 1, height, width)
        offset: torch.Tensor, size of (batch_size, 2, hegith, width)
        depth: torch.Tensor, size of (batch_size, 1, height, width)
        rotate: torch.Tensor, size of (batch_size, 4, height, width)

    Returns
    -------
    list of list of dict
        The dict has the following key-values.
        pitch: 
        yaw:
        roll:
        x:
        y:
        z:
        confidence:
    """
    heatmap = _sigmoid(output['heatmap'].detach())
    offset = output['offset'].detach()
    depth = 1. / (output['depth'].detach().sigmoid() + 1e-6) - 1
    quaternion = output['rotate'].detach()

    batch_size, _, height, width = heatmap.size()
    if on_nms:
        heatmap = _nms(heatmap)

    # TODO: compare topk and filtering by threshold
    scores, inds, classes, ys, xs = _topk(heatmap, K=k)
    offset = _transpose_and_gather_feat(offset, inds)
    offset = offset.view(batch_size, k, 2)
    # 2d-coordinate
    # (256*4, 256*5) --SCALE-> (2710, 3384)  SCALE = 2.64375
    xs = xs.view(batch_size, k, 1) + offset[:, :, 0:1]
    ys = ys.view(batch_size, k, 1) + offset[:, :, 1:2]
    xs = xs * config.MODEL_SCALE * config.SCALE
    ys = ys * config.MODEL_SCALE * config.SCALE
    # 3d-coordinate
    zs_world = _transpose_and_gather_feat(depth, inds)\
        .view(batch_size, k, 1)
    xs_world = (xs - CAMERA[0, 2]) * zs_world / CAMERA[0, 0]
    ys_world = (ys - CAMERA[1, 2]) * zs_world / CAMERA[1, 1]

    # calculate confidence, shape(batch_size, k) -> (batch_size, k, 1)
    confs = scores.view(batch_size, k, 1)

    # translate torch.Tensor to numpy.ndarray
    xyz_conf = torch.cat([xs_world, ys_world, zs_world, confs], dim=2)\
        .detach().cpu().numpy()

    # transform quaternion to euler angle
    # xyz = pitch yaw roll
    quaternion = _transpose_and_gather_feat(quaternion, inds)
    # shape(batch_size, k, 4) -> (batch_size * k, 4)
    quaternion = quaternion.view(-1, 4).detach().cpu().numpy()
    rotate = R.from_quat(quaternion).as_euler('xyz', degrees=False)
    rotate = rotate.reshape(batch_size, k, 3)

    # concatenate pitch, yaw, roll, x, y, z, confidence
    # shape of output is (batch_size, k, 7)
    preds = np.concatenate([rotate, xyz_conf], axis=2)

    # transform ndarray to list of list of dict
    label_list = ['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'confidence']
    pred_list = []
    for pred_points in preds:
        # pred_points: shape of (k, 7)
        # points_in_image: [{}, {}, ...]
        points_in_image = [dict(zip(label_list, pred_point))
                           for pred_point in pred_points]
        # points_in_image = clear_duplicates(points_in_image)
        pred_list.append(points_in_image)

    return pred_list


def clear_duplicates(points):
    for c1 in points:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in points:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < config.DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in points if c['confidence'] > 0]


def decode_train(heatmap, offset, depth, inds):
    batch_size, _, height, width = heatmap.size()

    offset = _transpose_and_gather_feat(offset, inds)
    offset = offset.view(batch_size, -1, 2)
    xs = xs.view(batch_size, -1, 1) + offset[:, :, 0:1]
    ys = ys.view(batch_size, -1, 1) + offset[:, :, 1:2]
    xs = xs * config.MODEL_SCALE
    ys = ys * config.MODEL_SCALE
    zs_world = _transpose_and_gather_feat(depth, inds)\
        .view(batch_size, k, 1)

    xs_world = (xs - CAMERA[0, 2]) * zs_world / CAMERA[0, 0]
    ys_world = (ys - CAMERA[1, 2]) * zs_world / CAMERA[1, 1]

    return xs_world, ys_world, zs_world

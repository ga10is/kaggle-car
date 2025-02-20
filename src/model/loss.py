import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _sigmoid(x):
    # y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * \
        neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='mean')

        return loss


class PosL1Loss(nn.Module):
    def __init__(self):
        super(PosL1Loss, self).__init__()

    def forward(self, output, target):
        """
        Parameters
        ----------
        output: size (b, 1, h, w)
        target: size (b, h, w)
        """
        target = target.unsqueeze(1)
        pos_ind = target.gt(0).float()
        num_pos = pos_ind.sum()
        loss = F.l1_loss(output * pos_ind, target * pos_ind, reduction='none')

        loss = loss.sum() / num_pos

        return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class QuatLoss(nn.Module):
    def __init__(self):
        super(QuatLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()

        print('pred', pred.size())

        # Quaternion diff = Quaternion.Normalize(q1) *
        #     Quaternion.Inverse(Quaternion.Normalize(q2));
        # shape (batch, 50, 4)?
        norm_pred = torch.norm(pred * mask, dim=2)
        norm_target = torch.norm(target * mask, dim=2)
        # inverse
        norm_target[:, :, :3] *= -1
        inv_target = norm_target
        mat = norm_pred * inv_target
        # w shape (batch, 50)
        w = -mat[:, :, 0] - mat[:, :, 1] - mat[:, :, 2] + mat[:, :, 3]

        # shape (batch,)
        n_mask = mask.squeeze().sum(dim=1)
        w = w.sum(dim=1) / n_mask

        loss = w.mean()

        return loss


class CarLoss(nn.Module):
    def __init__(self):
        super(CarLoss, self).__init__()
        self.crit_heatmap = FocalLoss()
        self.crit_hm_reg = PosL1Loss()
        self.crit_reg = L1Loss()
        self.crit_rot = L1Loss()

    def forward(self, outputs, data):
        """
        Parameters
        """
        loss_heatmap, loss_depth, loss_offset, loss_rotate = 0, 0, 0, 0
        loss_heatmap_reg = 0
        num_stacks = len(outputs)
        for s in range(num_stacks):
            output = outputs[s]
            # heatmap loss
            heatmap = _sigmoid(output['heatmap'])
            depth = 1. / (output['depth'].sigmoid() + 1e-6) - 1

            loss_heatmap += \
                self.crit_heatmap(heatmap, data['heatmap']) / num_stacks
            loss_heatmap_reg += self.crit_hm_reg(heatmap,
                                                 data['heatmap']) / num_stacks

            # depth loss
            loss_depth += self.crit_reg(
                depth, data['reg_mask'].long(), data['index'].long(), data['depth']) / num_stacks
            loss_offset += self.crit_reg(
                output['offset'], data['rot_mask'].long(), data['index'].long(), data['offset']) / num_stacks
            loss_rotate += self.crit_rot(
                output['rotate'], data['rot_mask'].long(), data['index'].long(), data['rotate']) / num_stacks

        loss = config.HM_WEIGHT * loss_heatmap \
            + config.HM_REG_WEIGHT * loss_heatmap_reg \
            + config.OFFSET_WEIGHT * loss_offset \
            + config.DEPTH_WEIGHT * loss_depth \
            + config.ROTATE_WEIGHT * loss_rotate
        loss_stats = {
            'loss': loss,
            'loss_heatmap': loss_heatmap,
            'loss_heatmap_reg': loss_heatmap_reg,
            'loss_offset': loss_offset,
            'loss_depth': loss_depth,
            'loss_rotate': loss_rotate,
        }

        return loss, loss_stats

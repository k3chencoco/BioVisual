import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
import random
import os


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes[1],
                        number_of_classes[0],
                        filter_size,
                        filter_size,), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes[1]):
        for j in range(number_of_classes[0]):
            weights[i,j, :, :] = upsample_kernel
    return torch.Tensor(weights)

def get_padding(output_size, input_size, factor):
    TH = output_size[2] - ((input_size[2]-1)*factor) - (factor*2)
    TW = output_size[3] - ((input_size[3]-1)*factor) - (factor*2)
    padding_H = int(np.ceil(TH / (-2)))
    out_padding_H = TH - padding_H*(-2)

    padding_W = int(np.ceil(TW / (-2)))
    out_padding_W = TW - padding_W*(-2)
    return (padding_H, padding_W), (out_padding_H, out_padding_W)

def cfgs2name(cfgs):
    name = '%s_%s_%s(%s,%s,%s)' % \
            (cfgs['dataset'], cfgs['backbone'], cfgs['loss'], cfgs['a'], cfgs['b'],cfgs['c'])
    if 'MultiCue' in cfgs['dataset']:
        name = name + '_' + str(cfgs['multicue_seq'])
    return name

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        #self.sum += val * n
        self.sum = self.sum+val*n
        self.count =self.count+ n
        self.avg = self.sum / self.count


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
        #    module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            module.weight = nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.neg_slope)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def adjust_learning_rate(optimizer, epoch, method, epochs, lr, lr_steps):
    if method == 'cosine':
        T_total = float(epochs)
        T_cur = float(epoch)
        lr = 0.5 * lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = lr
        for epoch_step in lr_steps:
            if epoch >= epoch_step:
                lr = lr * 0.1
    optimizer.lr = lr
    str_lr = '%.6f' % lr
    return str_lr


# class Cross_Entropy_Loss_RCF(nn.Module):
#     def __init__(self):
#         super(Cross_Entropy_Loss_RCF, self).__init__()
#     def forward(self, pred, labels):
#         pred_flat = pred.view(-1)
#         labels_flat = labels.view(-1)
#         pred_pos = pred_flat[labels_flat > 0]
#         pred_neg = pred_flat[labels_flat == 0]
#         total_loss =  cross_entropy_per_image(pred, labels, 1) +  0.1 * 0 * dice_loss_per_image(pred, labels)
#
#         return total_loss, (1-pred_pos).abs(), pred_neg

class Cross_Entropy_Loss_RCF(nn.Module):
    def __init__(self):
        super(Cross_Entropy_Loss_RCF, self).__init__()
    def forward(self, pred, labels):
        label = labels.long()
        mask = label.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()
        num_two = torch.sum((mask == 0.2).float()).float()
        # assert num_negative + num_positive + num_two == label.shape[0] * label.shape[1] * label.shape[2] * label.shape[3]
        # assert num_two == 0
        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 0.2] = 0
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]

        cost = F.binary_cross_entropy(
            pred.float(), label.float(), weight=mask,  reduction='sum')
        return cost, (1-pred_pos).abs(), pred_neg

class Cross_Entropy_Loss_Mod(nn.Module):
    def __init__(self):
        super(Cross_Entropy_Loss_Mod, self).__init__()
    def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]
        total_loss =  1.0 * cross_entropy_per_image(pred, labels, 0) + 0.3 * dice_loss_per_image(pred, labels)

        return total_loss, (1-pred_pos).abs(), pred_neg

class Cross_Entropy_Loss_PreUAED(nn.Module):
    def __init__(self):
        super(Cross_Entropy_Loss_PreUAED, self).__init__()
    def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]
        total_loss =  1.0 * cross_entropy_per_image(pred, labels, 3) +    1 * dice_loss_per_image(pred, labels)

        return total_loss, (1-pred_pos).abs(), pred_neg

#交叉熵
class BCELoss(nn.Module):
    def __init__(self, reduction="mean", pos_weight=1.0):
        pos_weight = torch.tensor(pos_weight).cuda()
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(
            reduction=reduction)

    def forward(self, prediction, targets):
        mask = targets.float()
        return self.bce_loss(prediction, targets)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.sigmoid(prediction)
        intersection = 2 * torch.sum(prediction * target) + self.smooth
        union = torch.sum(prediction) + torch.sum(target) + self.smooth
        loss = 1 - intersection / union
        return loss

class BCE_DiceLoss(nn.Module):
    def __init__(self, reduction="mean", B_weight=1,D_weight=1):
        super(BCE_DiceLoss, self).__init__()
        self.DiceLoss = DiceLoss()
        self.BCELoss = BCELoss(reduction=reduction)
        self.B_weight = B_weight
        self.D_weight = D_weight

    def forward(self, prediction, targets):
        pred_flat = prediction.view(-1)
        labels_flat = targets.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]
        return (self.D_weight * self.DiceLoss(prediction, targets) + self.B_weight * self.BCELoss(prediction,
                                                                                                       targets),
                (1-pred_pos).abs(), pred_neg)

class Cross_Entropy(nn.Module):
    def __init__(self):
        super(Cross_Entropy, self).__init__()
        # self.weight1 = nn.Parameter(torch.Tensor([1.]))
        # self.weight2 = nn.Parameter(torch.Tensor([1.]))

    def forward(self, pred, labels, side_output=None):
        # def forward(self, pred, labels):
        pred_flat = pred.view(-1)
        labels_flat = labels.view(-1)
        pred_pos = pred_flat[labels_flat > 0]
        pred_neg = pred_flat[labels_flat == 0]

        total_loss =  cross_entropy_per_image(pred, labels)+ 0.1*0 * dice_loss_per_image(pred, labels)
        if side_output is not None:
            for s in side_output:
                total_loss += cross_entropy_per_image(s, labels) / len(side_output)

        # total_loss = cross_entropy_per_image(pred, labels)
        # total_loss = dice_loss_per_image(pred, labels)
        # total_loss = 1.00 * cross_entropy_per_image(pred, labels) + \
        # 0.00 * 0.1 * dice_loss_per_image(pred, labels)
        # total_loss = self.weight1.pow(-2) * cross_entropy_per_image(pred, labels) + \
        #              self.weight2.pow(-2) * 0.1 * dice_loss_per_image(pred, labels) + \
        #              (1 + self.weight1 * self.weight2).log()
        return total_loss, (1-pred_pos).abs(), pred_neg

def dice(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    dice = ((logits * labels).sum() * 2 + eps) / (logits.sum() + labels.sum() + eps)
    dice_loss = 1-dice
    # dice_loss = 1 - dice
    return dice_loss


def dice_loss_per_image(logits, labels):
    total_loss = 0.0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += dice(_logit, _label)
    return total_loss / len(logits)

def cross_entropy_per_image(logits, labels, RCF=0):
    total_loss = 0.0
    for i, (_logit, _label) in enumerate(zip(logits, labels)):
        total_loss += cross_entropy_with_weight(_logit, _label, RCF)
    return total_loss / len(logits)

def cross_entropy_with_weight(logits, labels, RCF=0):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    if labels.sum() == 0:
        pred_neg = logits.clamp(eps, 1.0 - eps)
        return (-(1.0 - pred_neg).log()).mean()
    if labels.sum() == labels.numel():
        pred_pos = logits.clamp(eps, 1.0 - eps)
        return (-pred_pos.log()).mean()
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)
    if 1==RCF:
        pred_pos = logits[labels ==1].clamp(eps, 1.0 - eps)
        weight_pos, weight_neg = get_weight(labels, labels, 1.1)
        cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
                        (-(1.0 - pred_neg).log() * weight_neg).sum()
    elif 0==RCF:
        pred_pos = logits[labels > 0].clamp(eps,1.0 - eps)
        pred_neg = logits[labels == 0].clamp(eps,1.0-eps)
        w_anotation = labels[labels > 0]
        cross_entropy = (-pred_pos.log() * w_anotation).mean() + \
                    (-(1.0 - pred_neg).log()).mean()
    elif 2==RCF:
        pred_pos = logits[labels > 0].clamp(eps, 1.0 - eps)
        w_anotation = labels[labels > 0]
        cross_entropy = (-pred_pos.log() * w_anotation ).mean() + \
                        (-(1.0 - pred_neg).log() ).mean()

    else:
        pred_pos = logits[labels > 0].clamp(eps, 1.0 - eps)
        w_anotation = labels[labels > 0]
        proportion_neg, proportion_pos, count_neg, count_pos  = get_proportion(labels, labels, 1.1)
        mask = labels.float()
        mask[mask>0] = proportion_neg
        mask[mask==0] = proportion_pos
        cross_entropy = (-pred_pos.log() * w_anotation ).mean() + \
                        (-(1.0 - pred_neg).log() ).mean()

    return cross_entropy

def get_weight(src, mask, weight):
    count_pos = src[mask == 1].size()[0]
    count_neg = src[mask == 0].size()[0]
    total = count_neg + count_pos
    weight_pos = count_neg / total
    weight_neg = (count_pos / total) * weight
    return weight_pos, weight_neg

def get_proportion(src, mask, weight):
    count_pos = src[mask > 0].size()[0]
    count_neg = src[mask == 0].size()[0]
    total = count_neg + count_pos
    weight_neg = count_neg / total
    weight_pos = (count_pos / total) * weight
    return weight_neg, weight_pos, count_neg, count_pos

def get_count(src, mask):
    count_pos = src[mask > 0].size()[0]
    count_neg = src[mask == 0].size()[0]
    return count_pos, count_neg

def cross_entropy_orignal(logits, labels):
    logits = logits.view(-1)
    labels = labels.view(-1)
    eps = 1e-6
    pred_pos = logits[labels >= 0.5].clamp(eps, 1.0 - eps)
    pred_neg = logits[labels == 0].clamp(eps, 1.0 - eps)

    weight_pos, weight_neg = get_weight(labels, labels, 0.4, 1.5)

    cross_entropy = (-pred_pos.log() * weight_pos).sum() + \
                            (-(1.0 - pred_neg).log() * weight_neg).sum()
    return cross_entropy



def learning_rate_decay(optimizer, epoch, decay_rate=0.1, decay_steps=10):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (decay_rate ** (epoch // decay_steps))

def bilinear_upsample_weights(factor, number_of_classes):
    filter_size = 2 * factor - factor % 2
    weights = np.zeros((number_of_classes,
                        number_of_classes,
                        filter_size,
                        filter_size,), dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel
    return torch.Tensor(weights)
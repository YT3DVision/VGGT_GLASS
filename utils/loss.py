from utils import pytorch_ssim
from utils import pytorch_iou
import torch.nn as nn
import torch.nn.functional as F
import torch

# from utils.perceptual import  *
# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_iou_loss(pred, target):
    target = F.interpolate(target, size=(pred.size(-2), pred.size(-1)), mode="bilinear")
    bce_out = bce_loss(pred, target)
    # ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    # loss = bce_out + ssim_out + iou_out
    loss = bce_out + iou_out

    return loss


# only ssim loss
def only_ssim_loss(pred, target):
    ssim_out = 1 - ssim_loss(pred, target)

    return ssim_out


# only iou loss
def only_iou_loss(pred, target):
    iou_out = iou_loss(pred, target)

    return iou_out


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predictive, target, ep=1e-8):
    target = F.interpolate(target, size=(predictive.size(-2), predictive.size(-1)), mode="bilinear")
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    edge_loss = 1 - intersection / union
    bce_out = bce_loss(predictive, target)
    loss = edge_loss + bce_out
    return loss

# def perceptual_loss(pred, target):
# 	vgg_model = vgg16(pretrained=True).features[:16]
# 	vgg_model = vgg_model.cuda()
# 	for param in vgg_model.parameters():
# 		param.requires_grad = False
# 	loss_network = LossNetwork(vgg_model)
# 	loss_network.eval()
# 	perceptual_out = loss_network(pred, target)
#
# 	return  perceptual_out

# check the usage of loss

# import torch
# a = torch.randn(2,3,512,512).cuda()
# b = torch.randn(2,3,512,512).cuda()
# loss = perceptual_loss(a, b)
# loss2 = only_ssim_loss(a,b)
# print(loss2)
# print(loss)

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as matimg
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable


def iou_mean(pred, target, n_classes=1):
    # n_classes ：the number of classes in your dataset,not including background
    # for mask and ground-truth label, not probability map
    ious = []
    iousSum = 0
    pred = np.array(pred)
    pred = torch.from_numpy(pred)
    pred = pred.view(-1)
    target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(1, n_classes + 1):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / float(max(union, 1)))

            iousSum += float(intersection) / float(max(union, 1))
        # print(float(intersection) / float(max(union, 1)))
    return iousSum / n_classes



'''
# 
# if __name__ == "__main__":
# 
#     gt_dir = 'F:/GmdNetGroup/code/DailyMirrorSeg/try/9.png'
#     pred_dir = 'F:/GmdNetGroup/code/DailyMirrorSeg/try/9_r.png'
#     feat = matimg.imread(pred_dir)
#     true_mask = matimg.imread(gt_dir)
# 
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     index = torch.round(transform(feat)).squeeze(0).int()
# 
#     true_mask = torch.round(transform(true_mask)).squeeze(0).int()
# 
#     result = iou_mean(index, true_mask, 1)
#     print(result)
# 
# 
# import torch
# 
# 
# def mask_iou(mask1, mask2):
#     """
#     mask1: [m1,n] m1 means number of predicted objects
#     mask2: [m2,n] m2 means number of gt objects
#     Note: n means image_w x image_h
#     """
#     intersection = torch.matmul(mask1, mask2.t())
#     print(intersection)
#     area1 = torch.sum(mask1, dim=1).view(1, -1)
#     area2 = torch.sum(mask2, dim=1).view(1, -1)
#     union = (area1.t() + area2) - intersection
#     print(union)
#     iou = intersection / union
#     return iou
# 
# a = torch.randn(5,8) #均值为0方差为1的正态分布
# a.gt_(0) #二值化：大于0的数替换为1 小于0的数替换为0
# b = torch.randn(5,8)
# b.gt_(0)
# 
# 
# print(a)
# print(b)
# 
# print(mask_iou(a, b))
'''


'''
import torch
import numpy as np


def fast_hist(a, b, n):
    """
    生成混淆矩阵
    a 是形状为(HxW,)的预测值
    b 是形状为(HxW,)的真实值
    n 是类别数
    """
    # 确保a和b在0~n-1的范围内，k是(HxW,)的True和False数列
    # a = a.numpy()
    # b = b.numpy()
    k = (a >= 0) & (a < n)
    # 横坐标是预测的类别，纵坐标是真实的类别
    return torch.bincount(a[k] + n * b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    """
    hist传入混淆矩阵(n, n)
    """
    # 因为下面有除法，防止分母为0的情况报错
    np.seterr(divide="ignore", invalid="ignore")
    # 交集：np.diag取hist的对角线元素
    # 并集：hist.sum(1)和hist.sum(0)分别按两个维度相加，而对角线元素加了两次，因此减一次
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # 把报错设回来
    np.seterr(divide="warn", invalid="warn")
    # 如果分母为0，结果是nan，会影响后续处理，因此把nan都置为0
    iou[np.isnan(iou)] = 0.
    return iou


def get_MIoU(iou):
    return np.mean(iou)


a = torch.randn(5,8) #均值为0方差为1的正态分布
a.gt_(0) #二值化：大于0的数替换为1 小于0的数替换为0
b = torch.randn(5,8)
b.gt_(0)

print(fast_hist(a,b,2))
'''
import argparse

import cv2
from torch.autograd import Variable
from torchvision import transforms
import torch
import numpy as np
import time
import os
from utils.Miou import *
from tqdm import tqdm
from vggt.models.vggt_S import VGGT
from vggt.models.lora import apply_lora, load_lora
from utils.dataloader import *

scale = 384

#
parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
parser.add_argument("--data_path", type=str, default="E:/dataset/GSD", help="")
parser.add_argument("--save_path", type=str, default="./ckpt/vggt_s_backbone", help="")
parser.add_argument("--data_size", type=int, default=518, help="")
parser.add_argument("--result_path", type=str, default="./vggt_sv4_GSD_depth", help="")
parser.add_argument("--len_s", type=int, default=8, help="Training sequence length")

opt = parser.parse_args()

os.makedirs(opt.result_path, exist_ok=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

img_transform = transforms.Compose(
    [transforms.Resize((scale, scale)), transforms.ToTensor()]
)

target_transform = transforms.Compose(
    [transforms.Resize((scale, scale)), transforms.ToTensor()]
)

to_pil = transforms.ToPILImage()


def main():
    # ######## create Model #############
    model = VGGT(enable_camera=False, enable_point=True, enable_depth=True, enable_track=False, enable_glass=True).cuda().train()
    apply_lora(model)
    model.load_state_dict(torch.load("./ckpt/vggt_s_w_dpv4/vggt_s_basic.pth"), strict=False)
    # load_lora(model, os.path.join(opt.save_path, "lora.pth"))
    # model.glass_head.load_state_dict(torch.load(os.path.join(opt.save_path, "decoder.pth")))
    val_dataset = make_single_dataset(opt.data_path, data_size=opt.data_size, train=False)
    valid_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )
    # print params
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))

    model.eval()
    with torch.no_grad():
        start = time.time()
        v_glass_iou = 0
        val_iterator = tqdm(valid_loader)
        for idx, (glass_images, gt_images, edge_images, save_name) in enumerate(val_iterator):
            glass_images = glass_images.cuda()
            gt_images = gt_images.cuda()

            predictions = model(glass_images)
            depth = predictions['depth']
            point = predictions['world_points']
            # depth = depth.squeeze(0)
            # depth = depth.squeeze(0).detach().cpu().numpy()
            # point = point.squeeze(0)
            # point = point.squeeze(0).detach().cpu().numpy()
            # np.save(f"E:/dataset/all_single_glass/train/depth/{save_name[0].replace('.png', '.npy')}", depth)
            # np.save(f"E:/dataset/all_single_glass/train/point/{save_name[0].replace('.png', '.npy')}", point)
            # print(1)
            b, s, _, _, _ = depth.shape
            pred = depth.permute(0, 1, 4, 2, 3)
            pred.squeeze(2)
            for i in range(b):
                for j in range(s):
                    depth = pred[i, j, :, :]
                    depth = depth.squeeze(0).detach().cpu().numpy()
                    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

                    # 归一化到0~255
                    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                    depth_uint8 = depth_norm.astype(np.uint8)
                    cv2.imwrite(os.path.join(opt.result_path, save_name[i]), depth_uint8)
            # label = gt_images
            # b, s, _, _, _ = label.shape
            #
            # temp1 = pred.data.squeeze(2)
            # temp2 = label.data.squeeze(2)
            # for i in range(b):
            #     for j in range(s):
                    # a = temp1[i, j, :, :]
                    # b = temp2[i, j, :, :]
                    # a = torch.round(a).squeeze(0).int().detach().cpu()
                    # b = torch.round(b).squeeze(0).int().detach().cpu()
                    # v_glass_iou += iou_mean(a, b, 1)
                    # g_final = np.array(transforms.Resize((opt.data_size, opt.data_size))(to_pil(a)))
                    # cv2.imwrite(os.path.join(opt.result_path, save_name), g_final * 255.0)


        end = time.time()
        print("Average Time Is : {:.6f}".format((end - start) / len(valid_loader)))
        # print("Test IoU is : {:.2f}".format(v_glass_iou / (len(valid_loader)) * 100))


main()

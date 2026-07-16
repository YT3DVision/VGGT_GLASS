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
from vggt.models.vggt import VGGT
from vggt.models.lora import apply_lora, load_lora
from utils.dataloader import *

scale = 384

#
parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
parser.add_argument("--data_path", type=str, default="/data/huanghao/FP_VGD", help="")
parser.add_argument("--save_path", type=str, default="./lora", help="")
parser.add_argument("--data_size", type=int, default=518, help="")
parser.add_argument("--result_path", type=str, default="./vggt_lora", help="")
parser.add_argument("--len_s", type=int, default=8, help="Training sequence length")

opt = parser.parse_args()


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
    model = VGGT(enable_camera=False, enable_point=False, enable_depth=False, enable_track=False).cuda().train()
    apply_lora(model)
    model.load_state_dict(torch.load(os.path.join(opt.save_path, "validation_iou_max.pth")))
    val_dataset = make_dataSet(opt.data_path, data_size=opt.data_size, sequence_length=opt.len_s, train=False)
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
        for idx, (glass_images, gt_images, video_name, index) in enumerate(val_iterator):
            glass_images = glass_images.cuda()
            gt_images = gt_images.cuda()

            predictions = model(glass_images)
            pred = predictions['glass']
            label = gt_images
            b, s, _, _, _ = label.shape

            temp1 = pred.data.squeeze(2)
            temp2 = label.data.squeeze(2)
            for i in range(b):
                for j in range(s):
                    a = temp1[i, j, :, :]
                    b = temp2[i, j, :, :]
                    a = torch.round(a).squeeze(0).int().detach().cpu()
                    b = torch.round(b).squeeze(0).int().detach().cpu()
                    v_glass_iou += iou_mean(a, b, 1)
                    g_final = np.array(transforms.Resize((opt.data_size, opt.data_size))(to_pil(a)))
                    cv2.imwrite(os.path.join(opt.result_path, f'{video_name[i]}_{index[i]+j:08}.png'), g_final * 255.0)


        end = time.time()
        print("Average Time Is : {:.6f}".format((end - start) / len(valid_loader)))
        print("Test IoU is : {:.2f}".format(v_glass_iou / (3 * len(valid_loader)) * 100))


main()

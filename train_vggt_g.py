import argparse
import torch.optim as optim
import torch
import time
import os
from vggt.models.vggt_S import VGGT
from torch.utils.data import DataLoader
from utils.Miou import iou_mean
from utils.loss import bce_iou_loss, dice_loss
from utils.dataloader import *
import warnings
from torchvision import transforms as T
from tqdm import tqdm
from vggt.models.lora import apply_lora, save_lora, load_lora, merge_lora

scale = 384
x = torch.linspace(start=-1, end=1, steps=scale)
y = torch.linspace(start=-1, end=1, steps=scale)
G = torch.stack(torch.meshgrid(x, y)).unsqueeze(0).permute(0, 3, 2, 1).cuda()


def get_transform():
    return T.Compose([
        T.Resize((scale, scale)),
        T.ToTensor(),
    ])


def back_warp(image, light_flow):
    G1 = G.repeat(image.shape[0],1,1,1)
    wrap_image = torch.nn.functional.grid_sample(image, G1 + light_flow.permute(0, 2, 3, 1) / scale)
    return wrap_image


def loss_lt(N_1_pred, N_pred, N_1, N, light_flow):
    N_1_pred_w = back_warp(N_pred, light_flow)
    N_1_w = back_warp(N, light_flow)
    v = torch.exp(-1.0*torch.abs(N_1-N_1_w))
    q = N_1_pred - N_1_pred_w
    q = q*q
    return torch.mean(v*q)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    train_min_loss = float("inf")
    valid_max_glass = float(0)

    parser = argparse.ArgumentParser(description="PyTorch Glass Detection Example")
    parser.add_argument("--batchsize", type=int, default=4, help="Training batch size")
    parser.add_argument("--len_s", type=int, default=1, help="Training sequence length")
    parser.add_argument("--epochs", type=int, default=200, help="")
    parser.add_argument("--data_size", type=int, default=518, help="")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
    parser.add_argument("--data_path", type=str, default=r"E:/dataset/GDD", help="")
    parser.add_argument("--save_path", type=str, default="./ckpt/vggt_g_gdd", help="")
    parser.add_argument("--lr", type=float, default=1e-5, help="")

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    # load dataset
    print("INFO:Loading dataset ...\n")
    # train_loader, valid_loader, train_ds, val_ds = getMMD()
    train_dataset = make_single_dataset(opt.data_path, data_size=opt.data_size, train=True)
    val_dataset = make_single_dataset(opt.data_path, data_size=opt.data_size, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batchsize,
        num_workers=0,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        val_dataset,
        batch_size=opt.batchsize,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )
    print("# of training samples: %d\n" % int(len(train_loader)))
    print("# of valid samples: %d\n" % int(len(valid_loader)))

    model = VGGT(enable_camera=False, enable_point=True, enable_depth=True, enable_track=False).cuda().train()
    apply_lora(model)
    model.load_state_dict(torch.load("./pre_train/model.pth"), strict=False)
    merge_lora(model)
    # model.glass_head.load_state_dict(torch.load(os.path.join(opt.save_path, "decoder.pth")))

    # load_lora(model, os.path.join(opt.save_path, "lora.pth"))
    train_params = []
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            train_params.append(param)
        if 'glass_head' in name:
            param.requires_grad = True
            train_params.append(param)
    # state = torch.load(r"./model_res_backbone/validation_iou_max.pth", weights_only=True)
    # model.load_state_dict(state)
    # print("model initiating success")

    # print params
    total = sum(p.numel() for p in model.parameters())
    train_count = sum(p.numel() for p in train_params)
    print("Total params: %.2fM" % (total / 1e6))
    print("Train params: %.2fM" % (train_count / 1e6))

    # initiate optimizer
    optimizer = optim.Adam(train_params, lr=opt.lr)
    device = torch.device('cuda')
    for epoch in range(opt.epochs):
        start = time.time()
        model.train()
        model.zero_grad()

        train_loss_sum = 0
        t_glass_iou = 0
        iters = 0
        train_iterator = tqdm(train_loader, total=len(train_loader))
        for i, (glass_images, gt_images, edge_images, save_name) in enumerate(train_iterator):
            # data_time = time.time()
            glass_images = glass_images.cuda()
            gt_images = gt_images.cuda()
            edge_images = edge_images.cuda()
            b, s, c, h, w = gt_images.shape
            # print(h, w)
            optimizer.zero_grad()
            predictions = model(glass_images)
            # run_time = time.time()-data_time
            glass_list = predictions['glass']
            edge_list = predictions['edge']
            gt_images = gt_images.reshape(b * s, c, h, w)
            edge_images = edge_images.reshape(b * s, c, h, w)
            alpha = 1
            loss_glass = 0
            loss_edge = 0
            for i in range(len(glass_list)):
                glass_list[i] = glass_list[i].reshape(b * s, c, h, w)
                loss_glass = loss_glass+alpha*bce_iou_loss(glass_list[i], gt_images)
                edge_list[i] = edge_list[i].reshape(b * s, c, h, w)
                loss_edge = loss_edge + alpha*dice_loss(edge_list[i], edge_images)
                alpha = alpha / 2
            loss = loss_glass + loss_edge
            # print(loss)
            loss.backward()
            optimizer.step()
            # loss_time = time.time()-run_time-data_time
            train_loss_sum += loss.item()
            iters += 1
            # print('run_time:', run_time, 'loss_time:', loss_time)

        model.eval()
        v_glass_iou = 0
        with torch.no_grad():
            val_iterator = tqdm(valid_loader)
            for i, (glass_images, gt_images, edge_images, save_name) in enumerate(val_iterator):
                glass_images = glass_images.cuda()
                gt_images = gt_images.cuda()

                predictions = model(glass_images)

                pred = predictions['glass'][0]
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

                torch.cuda.empty_cache()

        end = time.time()
        t = end - start

        print(
            "INFO: epoch:{},train loss:{},validation iou:{}, time:{}".format(
                epoch + 1, train_loss_sum / len(train_loader), (v_glass_iou / (len(valid_loader)*opt.batchsize)) * 100,
                round(t, 2),
            )
        )
        # if train_loss_sum < train_min_loss:
        #     train_min_loss = train_loss_sum
        #     torch.save(model.glass_head.state_dict(), os.path.join(opt.save_path, "train_loss_min.pth"))
        #     print("INFO: save train_loss_min model")
        if v_glass_iou > valid_max_glass:
            valid_max_glass = v_glass_iou
            torch.save(model.glass_head.state_dict(), os.path.join(opt.save_path, "decoder.pth"))
            save_lora(model, os.path.join(opt.save_path, "lora.pth"))
            print("INFO: save validation_iou_max model")

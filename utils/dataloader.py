import os
import os.path
import torch.utils.data as data
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import torchvision.transforms.functional as TF
from vggt.utils.load_fn import load_and_preprocess_images


def make_data(data_path, length, train):
    if train:
        data_path = os.path.join(data_path, 'train')
        print('INFO: Processing Train Data')
    else:
        data_path = os.path.join(data_path, 'test')
        print('INFO: Processing Test Data')
    img_list = os.listdir(data_path)
    results = []
    for video_name in img_list:
        frame_list = [f for f in os.listdir(os.path.join(data_path, video_name, 'JPEGImages')) if f.endswith('.jpg')]
        for index in range(0, len(frame_list)-length+1, length):
            glass_list = []
            gt_list = []
            for k in range(length):
                glass_list.append(os.path.join(data_path, video_name, 'JPEGImages', f'{index+k:08}.jpg'))
                gt_list.append(os.path.join(data_path, video_name, 'SegmentationClassPNG', f'{index+k:08}.png'))
            results.append(
                (glass_list, gt_list, video_name, index)
            )
    return results


def make_video_as_single(data_path, train):
    if train:
        data_path = os.path.join(data_path, 'train')
        print('INFO: Processing Train Data')
    else:
        data_path = os.path.join(data_path, 'test')
        print('INFO: Processing Test Data')
    video_list = os.listdir(data_path)
    results = []
    for video_name in video_list:
        frame_list = [f for f in os.listdir(os.path.join(data_path, video_name, 'JPEGImages')) if f.endswith('.jpg')]

        for index in range(0, len(frame_list)):
            image_path = os.path.join(data_path, video_name, 'JPEGImages', f'{index:08}.jpg')
            gt_path = os.path.join(data_path, video_name, 'SegmentationClassPNG', f'{index:08}.png')
            edge_path = os.path.join(data_path, video_name, 'edge', f'{index:08}.png')
            save_name = video_name + f'_{index:08}.png'
            results.append(
                (image_path, gt_path, edge_path, save_name)
            )
    return results

def make_single_data(data_path, train):
    if train:
        data_path = os.path.join(data_path, 'train')
        print('INFO: Processing Train Data')
    else:
        data_path = os.path.join(data_path, 'test')
        print('INFO: Processing Test Data')
    image_list = [f[:-4] for f in os.listdir(os.path.join(data_path, 'image')) if f.endswith('.jpg')]
    results = []
    for image_name in image_list:
        image_path = os.path.join(data_path, 'image', image_name+'.jpg')
        gt_path = os.path.join(data_path, 'mask', image_name+'.png')
        edge_path = os.path.join(data_path, 'edge', image_name + '.png')
        save_name = image_name+'.png'
        results.append((image_path, gt_path, edge_path, save_name))
    return results


def make_union_data(data_path, train):
    if train:
        data_path = os.path.join(data_path, 'train')
        print('INFO: Processing Train Data')
    else:
        data_path = os.path.join(data_path, 'test')
        print('INFO: Processing Test Data')
    image_list = [f[:-4] for f in os.listdir(os.path.join(data_path, 'image')) if f.endswith('.jpg')]
    results = []
    for image_name in image_list:
        image_path = os.path.join(data_path, 'image', image_name+'.jpg')
        glass_path = os.path.join(data_path, 'mask', image_name+'.png')
        depth_path = os.path.join(data_path, 'depth', image_name+'.npy')
        point_path = os.path.join(data_path, 'point', image_name+'.npy')
        depth_fix_path = os.path.join(data_path, 'depth_fix', image_name+'.npy')
        point_fix_path = os.path.join(data_path, 'point_fix', image_name+'.npy')
        edge_path = os.path.join(data_path, 'edge', image_name + '.png')
        save_name = image_name+'.png'
        results.append((image_path, glass_path, edge_path, save_name, depth_path, point_path, depth_fix_path, point_fix_path))
    return results


def make_rgbd_data(data_path, train):
    if train:
        data_path = os.path.join(data_path, 'train')
        print('INFO: Processing Train Data')
    else:
        data_path = os.path.join(data_path, 'test')
        print('INFO: Processing Test Data')
    image_list = [f[:-4] for f in os.listdir(os.path.join(data_path, 'images')) if f.endswith('.jpg')]
    results = []
    for image_name in image_list:
        image_path = os.path.join(data_path, 'images', image_name+'.jpg')
        model_path = os.path.join(data_path, 'depths', image_name+'.png')
        gt_path = os.path.join(data_path, 'masks', image_name+'.png')
        save_name = image_name+'.png'
        results.append((image_path, model_path, gt_path, save_name))
    return results


def make_rgbt_data(data_path, train):
    if train:
        data_path = os.path.join(data_path, 'train')
        print('INFO: Processing Train Data')
    else:
        data_path = os.path.join(data_path, 'test')
        print('INFO: Processing Test Data')
    image_list = [f[:-4] for f in os.listdir(os.path.join(data_path, 'rgb')) if f.endswith('.png')]
    results = []
    for image_name in image_list:
        image_path = os.path.join(data_path, 'rgb', image_name+'.png')
        model_path = os.path.join(data_path, 'temperature_img', image_name.replace('rgb', 'temperature') + '.png')
        gt_path = os.path.join(data_path, 'mask', image_name.replace('rgb', 'mask')+'.png')
        save_name = image_name.replace('rgb', 'mask')+'.png'
        results.append((image_path, model_path, gt_path, save_name))
    return results

def make_mirror_data(data_path, train):
    results = []
    if train:
        video_path = os.path.join(data_path, 'frames')
        gt_path = os.path.join(data_path, 'masks')
        edge_path = os.path.join(data_path, 'edges')
        print('INFO: Processing Train Data')
    else:
        video_path = os.path.join(data_path, 'valid_frames')
        gt_path = os.path.join(data_path, 'valid_masks')
        edge_path = os.path.join(data_path, 'valid_edges')
        print('INFO: Processing Test Data')
    for video_name in os.listdir(video_path):
        frame_list = sorted([f for f in os.listdir(os.path.join(video_path, video_name)) if f.endswith('.png')])
        gt_list = sorted([f for f in os.listdir(os.path.join(gt_path, video_name.replace('frames', 'interpolated_set_'))) if f.endswith('.png')])
        edge_list = sorted([f for f in os.listdir(os.path.join(edge_path, video_name.replace('frames', 'interpolated_set_')+"_edges")) if f.endswith('.png')])
        for index in range(0, len(frame_list)-2, 1):
            results.append(
                (os.path.join(video_path, video_name, frame_list[index]),
                 os.path.join(video_path, video_name, frame_list[index+1]),
                 os.path.join(video_path, video_name, frame_list[index+2]),
                 os.path.join(gt_path, video_name.replace('frames', 'interpolated_set_'), gt_list[index]),
                 os.path.join(gt_path, video_name.replace('frames', 'interpolated_set_'), gt_list[index+1]),
                 os.path.join(gt_path, video_name.replace('frames', 'interpolated_set_'), gt_list[index+2]),
                 os.path.join(edge_path, video_name.replace('frames', 'interpolated_set_')+"_edges", edge_list[index]),
                 os.path.join(edge_path, video_name.replace('frames', 'interpolated_set_')+"_edges", edge_list[index+1]),
                 os.path.join(edge_path, video_name.replace('frames', 'interpolated_set_')+"_edges", edge_list[index+2]),
                 )
            )
    return results


def random_transform(input_img: Image.Image, target_img: Image.Image):
    # 随机水平翻转
    if random.random() > 0.5:
        input_img = TF.hflip(input_img)
        target_img = TF.hflip(target_img)

    # 随机垂直翻转
    if random.random() > 0.5:
        input_img = TF.vflip(input_img)
        target_img = TF.vflip(target_img)

    # 随机旋转（角度范围 -30 到 30）
    angle = random.uniform(-30, 30)
    input_img = TF.rotate(input_img, angle, fill=0)
    target_img = TF.rotate(target_img, angle, fill=0)
    w, h = input_img.size
    # 随机裁剪
    i, j, h, w = transforms.RandomCrop.get_params(input_img, output_size=(h//2, w//2))
    input_img = TF.crop(input_img, i, j, h, w)
    target_img = TF.crop(target_img, i, j, h, w)

    return input_img, target_img

class make_dataSet(data.Dataset):
    def __init__(self, data_path, data_size=384, sequence_length=6, train=True):
        self.train = train
        self.data_path = data_path
        self.length = sequence_length
        self.grey_transform = transforms.Compose([
        transforms.Resize((data_size, data_size)),
        transforms.ToTensor()
    ])
        self.rgb_transform = transforms.Compose([
        transforms.Resize((data_size, data_size)),
        transforms.ToTensor()
    ])
        self.images = make_data(data_path, self.length, self.train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        glass_list, gt_list, video_name, index = self.images[index]
        glass_images = []
        for image_path in glass_list:
            img = Image.open(image_path).convert("RGB")
            glass_images.append(self.rgb_transform(img))
        gt_images = []
        for gt_path in gt_list:
            gt = Image.open(gt_path).convert("L")
            gt_images.append(self.grey_transform(gt))
        glass_images = torch.stack(glass_images)
        gt_images = torch.stack(gt_images)
        # glass_images = load_and_preprocess_images(glass_list, mode="pad")
        # gt_images = load_and_preprocess_images(gt_list, mode="pad")
        # gt_images = gt_images.mean(dim=1, keepdim=True)
        return glass_images, gt_images, video_name, index

class make_single_dataset(data.Dataset):
    def __init__(self, data_path, data_size=384, train=True):
        self.train = train
        self.data_path = data_path
        self.grey_transform = transforms.Compose([
        transforms.Resize((data_size, data_size)),
        transforms.ToTensor()
    ])
        self.rgb_transform = transforms.Compose([
        transforms.Resize((data_size, data_size)),
        transforms.ToTensor()
    ])
        self.images = make_video_as_single(data_path, self.train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # glass_path, gt_path, edge_path, save_name, depth_path, point_path = self.images[index]
        glass_path, gt_path, edge_path, save_name = self.images[index]
        img = Image.open(glass_path).convert("RGB")
        img = self.rgb_transform(img)

        edge = Image.open(edge_path).convert("L")
        edge = self.grey_transform(edge)

        gt = Image.open(gt_path).convert("L")
        gt = self.grey_transform(gt)

        glass_images = img.unsqueeze(0)
        gt_images = gt.unsqueeze(0)
        edge_images = edge.unsqueeze(0)
        # glass_images = load_and_preprocess_images(glass_list, mode="pad")
        # gt_images = load_and_preprocess_images(gt_list, mode="pad")
        # gt_images = gt_images.mean(dim=1, keepdim=True)
        # if self.train:
        #     depth = torch.tensor(np.load(depth_path)).unsqueeze(0).squeeze(-1)
        #     point = torch.tensor(np.load(point_path)).unsqueeze(0)
        #     return glass_images, gt_images, edge_images, save_name, depth, point
        # else:
        return glass_images, gt_images, edge_images, save_name


class make_union_dataset(data.Dataset):
    def __init__(self, data_path, data_size=384, train=True):
        self.train = train
        self.data_path = data_path
        self.grey_transform = transforms.Compose([
        transforms.Resize((data_size, data_size)),
        transforms.ToTensor()
    ])
        self.rgb_transform = transforms.Compose([
        transforms.Resize((data_size, data_size)),
        transforms.ToTensor()
    ])
        self.images = make_union_data(data_path, self.train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        glass_path, gt_path, edge_path, save_name, depth_path, point_path, depth_fix_path, point_fix_path = self.images[index]
        # glass_path, gt_path, edge_path, save_name = self.images[index]
        img = Image.open(glass_path).convert("RGB")
        img = self.rgb_transform(img)

        edge = Image.open(edge_path).convert("L")
        edge = self.grey_transform(edge)

        gt = Image.open(gt_path).convert("L")
        gt = self.grey_transform(gt)

        glass_images = img.unsqueeze(0)
        gt_images = gt.unsqueeze(0)
        edge_images = edge.unsqueeze(0)
        # glass_images = load_and_preprocess_images(glass_list, mode="pad")
        # gt_images = load_and_preprocess_images(gt_list, mode="pad")
        # gt_images = gt_images.mean(dim=1, keepdim=True)
        if self.train:
            depth = torch.tensor(np.load(depth_path)).unsqueeze(0).squeeze(-1)
            point = torch.tensor(np.load(point_path)).unsqueeze(0)
            depth_fix = torch.tensor(np.load(depth_fix_path)).unsqueeze(0)
            point_fix = torch.tensor(np.load(point_fix_path)).unsqueeze(0)
            depth = [depth, depth_fix]
            point = [point, point_fix]

            return glass_images, gt_images, edge_images, save_name, depth, point
        else:
            return glass_images, gt_images, edge_images, save_name



class make_mutil_dataset(data.Dataset):
    def __init__(self, data_path, data_size=384, train=True):
        self.train = train
        self.data_path = data_path
        self.grey_transform = transforms.Compose([
            transforms.Resize((data_size, data_size)),
            transforms.ToTensor()
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize((data_size, data_size)),
            transforms.ToTensor()
        ])
        self.images = make_rgbt_data(data_path, self.train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        glass_path, model_path, gt_path, save_name = self.images[index]
        img = Image.open(glass_path).convert("RGB")
        img = self.rgb_transform(img)

        model = Image.open(model_path).convert("RGB")
        model = self.rgb_transform(model)

        gt = Image.open(gt_path).convert("L")
        gt = self.grey_transform(gt)

        glass_images = img.unsqueeze(0)
        model_images = model.unsqueeze(0)
        # input_images = torch.cat((glass_images, model_images), dim=0)
        input_images = glass_images
        gt_images = gt.unsqueeze(0)
        # glass_images = load_and_preprocess_images(glass_list, mode="pad")
        # gt_images = load_and_preprocess_images(gt_list, mode="pad")
        # gt_images = gt_images.mean(dim=1, keepdim=True)
        return input_images, gt_images, save_name












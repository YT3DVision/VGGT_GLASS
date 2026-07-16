import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from vggt.models.vggt_Sv2 import VGGT
from vggt.models.lora import apply_lora, load_lora


def numpy_to_torch(img_np, normalize=False, add_batch_dim=True):
    """将NumPy图像转换为PyTorch张量"""
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    pil_img = Image.fromarray(img_np)

    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
    transform = transforms.Compose(transform_list)
    img_tensor = transform(pil_img)

    if add_batch_dim:
        img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def generate_heatmap(feature_map, original_img):
    """生成热力图"""
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu().numpy()

    if feature_map.ndim == 4:
        feature_map = feature_map[0]

    # 聚合通道
    heatmap = np.mean(feature_map, axis=0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return heatmap_colored


class MultiFeatureExtractor:
    """提取指定层的多个特征"""
    def __init__(self, model, target_layer, indices):
        self.model = model
        self.target_layer = target_layer
        self.indices = indices
        self.features = {}
        self.hook = target_layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        with torch.no_grad():
            for idx in self.indices:
                feat = output[0][idx].detach().squeeze(0)
                feat = feat[:, 5:, :].view(1, 518 // 14, 518 // 14, -1).permute(0, 3, 1, 2)
                self.features[idx] = feat

    def remove(self):
        self.hook.remove()


class DPFeatureExtractor:
    """提取指定层的多个特征"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = {}
        self.hook = target_layer.register_forward_hook(self.save_features)

    def save_features(self, module, input, output):
        with torch.no_grad():
            feat = output[2].detach().squeeze(0)
            self.features[0] = feat

    def remove(self):
        self.hook.remove()

def process_folder(model, input_folder, output_folder):
    """批量处理文件夹下所有图片"""
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"❌ 无法读取: {img_name}")
            continue

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (518, 518))

        input_tensor = numpy_to_torch(original_img).cuda()

        # 输出文件夹（以图片名为子文件夹名）
        base_name = os.path.splitext(img_name)[0]
        save_dir = os.path.join(output_folder, base_name)
        os.makedirs(save_dir, exist_ok=True)

        # extractor = DPFeatureExtractor(model, model.aggregator)
        with torch.no_grad():
            predictions = model(input_tensor.cuda())
            feature_list = predictions["features"]
        # features = extractor.features
        # extractor.remove()

        #
        # depth = feature_list["depth_feature"][0][0]
        # heatmap = generate_heatmap(depth, original_img)
        # save_path = os.path.join(save_dir, f"depth_fea.png")
        # cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        #
        # point = feature_list["point_feature"][0][0]
        # heatmap = generate_heatmap(point, original_img)
        # save_path = os.path.join(save_dir, f"point_fea.png")
        # cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        backbone = feature_list["backbone"]
        for i in range(len(backbone)):
            backbone_fea = backbone[i].view(1, 518 // 14, 518 // 14, -1).permute(0, 3, 1, 2)
            heatmap = generate_heatmap(backbone_fea, original_img)
            save_path = os.path.join(save_dir, f"backbone_{i}.png")
            cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        fsam = feature_list["fsam_out"]
        for i in range(len(fsam)):
            heatmap = generate_heatmap(fsam[i], original_img)
            save_path = os.path.join(save_dir, f"fsam_{i}.png")
            cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        #
        # mmfb = feature_list["mmfb_out"]
        # for i in range(len(mmfb)):
        #     heatmap = generate_heatmap(mmfb[i], original_img)
        #     save_path = os.path.join(save_dir, f"mmfb_{i}.png")
        #     cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        #
        # decoder = feature_list["decoder_out"]
        # for i in range(len(decoder)):
        #     heatmap = generate_heatmap(decoder[i], original_img)
        #     save_path = os.path.join(save_dir, f"decoder_{i}.png")
        #     cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        # 保存每层的热图
        # for idx, feat in feature_list.items():
        #     heatmap = generate_heatmap(feat, original_img)
        #     save_path = os.path.join(save_dir, f"{idx}_point_fea.png")
        #     cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        print(f"已保存特征图: {img_name}")


if __name__ == "__main__":
    # ===== 配置路径 =====
    input_folder = "E:/dataset/GDD/test/image"     # 输入图像文件夹
    output_folder = "./feature_viz_GDD"            # 输出保存路径
    model_path = "./ckpt/vggt_s_v4_fdfp/vggt_s_basic.pth"            # 模型权重路径
    for image_name in os.listdir(output_folder):
        image = cv2.imread(os.path.join("E:/dataset/GDD/test/image", image_name+".jpg"))
        mask = cv2.imread(os.path.join("E:/dataset/GDD/test/mask", image_name+".png"))

        cv2.imwrite(os.path.join(output_folder, image_name, "rgb.png"), image)
        cv2.imwrite(os.path.join(output_folder, image_name, "gt.png"), mask)
    # ===== 加载模型 =====
    # model = VGGT(enable_camera=False, enable_point=True, enable_depth=True, enable_track=False).cuda().eval()
    # # apply_lora(model)
    # model.load_state_dict(torch.load(model_path), strict=False)
    # # apply_lora(model)
    # # load_lora(model, 'E:/LYW/vggt-main/lora_GDD/lora.pth')
    # # # ===== 处理整个文件夹 =====
    # process_folder(model, input_folder, output_folder)
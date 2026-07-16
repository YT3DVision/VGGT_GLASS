import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.glass_upsample4 = UpGlass(1024, 512)
        self.glass_upsample3 = UpGlass(512, 256)
        self.glass_upsample2 = UpGlass(256, 128)
        self.glass_upsample1 = UpGlass(128, 64)
        self.glass_upsample_final = UpGlass(64, 32)

        self.deconv4_glass = BasicConv(512, 512)
        self.deconv3_glass = BasicConv(256, 256)
        self.deconv2_glass = BasicConv(128, 128)
        self.deconv1_glass = BasicConv(64, 64)

        self.glass_pred_final = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, fea_list):
        feature_temp = self.glass_upsample4(fea_list[3])
        if feature_temp.size() != fea_list[2].size():
            feature_temp = F.interpolate(feature_temp, size=(fea_list[2].size(-2), fea_list[2].size(-1)), mode="bilinear")
        d4_glass = self.deconv4_glass(feature_temp + fea_list[2])

        feature_temp = self.glass_upsample3(d4_glass)
        if feature_temp.size() != fea_list[1].size():
            feature_temp = F.interpolate(feature_temp, size=(fea_list[1].size(-2), fea_list[1].size(-1)), mode="bilinear")
        d3_glass = self.deconv3_glass(feature_temp + fea_list[1])

        feature_temp = self.glass_upsample2(d3_glass)
        if feature_temp.size() != fea_list[0].size():
            feature_temp = F.interpolate(feature_temp, size=(fea_list[0].size(-2), fea_list[0].size(-1)), mode="bilinear")
        d2_glass = self.deconv2_glass(feature_temp + fea_list[0])

        feature_temp = self.glass_upsample1(d2_glass)
        d1_glass = self.deconv1_glass(feature_temp)

        feature_temp = self.glass_upsample_final(d1_glass)
        glass = self.glass_pred_final(feature_temp)

        return F.sigmoid(glass)


class UpGlass(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=4, stride=2, padding=1):
        super().__init__()
        self.up_glass = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, padding),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up_glass(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = DepthWiseConv(in_planes, out_planes)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.change = in_planes == out_planes
        self.skip_add = nn.Conv2d(in_planes, out_planes, 1)

    def forward(self, x):
        x1 = self.conv(x)
        if self.bn is not None:
            x1 = self.bn(x1)
        if self.relu is not None:
            x1 = self.relu(x1)
        if self.change:
            x1 = x + x1
        else:
            x1 = self.skip_add(x) + x1
        return x1


class GLASSHead(nn.Module):
    def __init__(self,
        dim_in: int,
        patch_size: int = 14,
        output_dim: int = 1,
        out_channels: List[int] = [128, 256, 512, 1024],
        intermediate_layer_idx: List[int] = [5, 11, 17, 23],
        ):
        super(GLASSHead, self).__init__()
        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        self.decode = SimpleDecoder()

    def forward(
            self,
            aggregated_tokens_list: List[torch.Tensor],
            images: torch.Tensor,
            patch_start_idx: int,
            frames_chunk_size: int = 8,
        ):
        B, S, _, H, W = images.shape
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            glass_preds = self._forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_preds.append(glass_preds)

        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ):
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        glass_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            x = x.reshape(B * S, -1, x.shape[-1])

            x = self.norm(x)

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[glass_idx](x)

            x = self.resize_layers[glass_idx](x)

            out.append(x)
            glass_idx += 1
        glass = self.decode(out)
        glass = F.interpolate(glass, size=(H, W), mode="bilinear")
        glass = glass.view(B, S, *glass.shape[1:])
        return glass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union
import math

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()

        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)

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
        image_size: int = 518,
        output_dim: int = 1,
        out_channels: List[int] = [128, 256, 512, 1024],
        intermediate_layer_idx: List[int] = [5, 11, 17, 23],
        ):
        super(GLASSHead, self).__init__()
        self.patch_size = patch_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.norm = nn.LayerNorm(dim_in)
        # self.reduce = nn.Linear(dim_in, dim_in//2)
        # self.memory_attention = Attention(dim_in)
        self.fs_block = nn.ModuleList(
            [FusionBlock(dim_in) for i in range(len(intermediate_layer_idx))]
        )
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
        self.cbam_bloack = nn.ModuleList(
            [CBAM(in_channels=oc) for oc in out_channels]
        )
        self.glass_head = nn.Linear(dim_in*2, 1)
        self.edge_head = nn.Linear(dim_in*2, 1)
        self.decode = Decoder(channel_list=out_channels)

    def forward(
            self,
            aggregated_tokens_list: List[torch.Tensor],
            f_dpt: torch.Tensor,
            f_p3d: torch.Tensor,
            images: torch.Tensor,
            patch_start_idx: int,
            frames_chunk_size: int = 8,
        ):
        B, S, _, H, W = images.shape
        f_dpt = f_dpt.reshape(B*S, -1, H, W)
        f_p3d = f_p3d.reshape(B*S, -1, H, W)
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, f_dpt, f_p3d, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0
        all_preds = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            glass_preds = self._forward_impl(
                aggregated_tokens_list, f_dpt, f_p3d, images, patch_start_idx, frames_start_idx, frames_end_idx
            )
            all_preds.append(glass_preds)

        return torch.cat(all_preds, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        f_dpt: torch.Tensor,
        f_p3d: torch.Tensor,
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
        h_memory = torch.ones(B*S, patch_h*patch_w, aggregated_tokens_list[0].shape[-1]).to(images.device)
        # self.glass_memory.to(images.device)
        out = []
        glass_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = x.reshape(B * S, -1, x.shape[-1])

            # x = self.memory_attention(x, self.glass_memory)

            x = self.norm(x)

            # x = self.reduce(x)
            out.append(x)
        c_glass = torch.sigmoid(self.glass_head(torch.cat((out[-2], out[-1]), dim=-1)))
        c_glass = c_glass.permute(0, 2, 1).reshape((c_glass.shape[0], c_glass.shape[-1], patch_h, patch_w))
        c_edge = torch.sigmoid(self.edge_head(torch.cat((out[0], out[1]), dim=-1)))
        c_edge = c_edge.permute(0, 2, 1).reshape((c_edge.shape[0], c_edge.shape[-1], patch_h, patch_w))
        for i in range(len(out)-1, -1, -1):
            out[i], h_memory = self.fs_block[i](out[i], h_memory)
            out[i] = out[i].permute(0, 2, 1).reshape((out[i].shape[0], out[i].shape[-1], patch_h, patch_w))

            out[i] = self.projects[i](out[i])

            out[i] = self.resize_layers[i](out[i])

            out[i] = self.cbam_bloack[i](out[i])

        # out = self.token_aggre(out)
        glass_list, edge_list = self.decode(out, c_glass, c_edge, f_dpt, f_p3d)
        glass_list.append(c_glass)
        edge_list.append(c_edge)
        for i in range(len(glass_list)):
            glass_list[i] = F.interpolate(glass_list[i], size=(H, W), mode="bilinear")
            glass_list[i] = glass_list[i].view(B, S, *glass_list[i].shape[1:])
            edge_list[i] = F.interpolate(edge_list[i], size=(H, W), mode="bilinear")
            edge_list[i] = edge_list[i].view(B, S, *edge_list[i].shape[1:])
        # glass = F.interpolate(glass, size=(H, W), mode="bilinear")
        # glass = glass.view(B, S, *glass.shape[1:])

        return glass_list, edge_list


class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x, y):
        B, N, D = x.shape
        _, s, _ = y.shape
        y.expand(B, s, D)
        Q = self.query(x)
        K = self.key(y)
        V = self.value(y)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V) + x
        output = self.norm(output)
        return output


class FusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_h = nn.Linear(dim, dim)
        self.linear_in = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, h_prev):
        h_i = self.linear_h(h_prev) + self.linear_in(x)
        h_i = self.norm(h_i)
        out = self.act(h_i) + x
        return out, h_i


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, channel, kernel_size=3, reduction=16, bias=True, act=nn.ReLU()):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2, bias=bias),
            act,
            nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2, bias=bias)
        )
        self.ca = CALayer(channel, reduction)

    def forward(self, x):
        res = self.body(x)
        res = self.ca(res)
        res += x
        return res


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        ca_out = self.channel_attention(x) * x
        sa_out = self.spatial_attention(ca_out) * ca_out
        return sa_out


class Decoder_block(nn.Module):
    def __init__(self, in_channel, out_channel, has_res=True):
        super(Decoder_block, self).__init__()
        self.has_res = has_res
        self.glass_upsample = UpGlass(in_channel=in_channel, out_channel=out_channel)
        self.proj = BasicConv(out_channel, out_channel)
        self.edge_conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        self.cbam = CBAM(out_channel)
        self.rcab = RCAB(out_channel)
        self.out_conv = nn.Conv2d(out_channel+128, out_channel, kernel_size=1)
        self.pred_glass = nn.Conv2d(out_channel, 1, kernel_size=1)
        self.pred_edge = nn.Conv2d(out_channel, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, fea_de, fea_en, fea_dep, fea_p3d, m_glass, m_edge):
        if self.has_res:
            res = self.glass_upsample(fea_de)
            res = F.interpolate(res, size=(fea_en.size(-2), fea_en.size(-1)), mode="bilinear")
            res = self.proj(res+fea_en)
        else:
            res = self.proj(fea_en)
        res_low = self.edge_conv(res)
        res_high = res - res_low
        fea_dep = F.interpolate(fea_dep, size=(fea_en.size(-2), fea_en.size(-1)), mode="bilinear")
        fea_p3d = F.interpolate(fea_p3d, size=(fea_en.size(-2), fea_en.size(-1)), mode="bilinear")
        m_glass = F.interpolate(m_glass, size=(fea_en.size(-2), fea_en.size(-1)), mode="bilinear")
        m_edge = F.interpolate(m_edge, size=(fea_en.size(-2), fea_en.size(-1)), mode="bilinear")
        fea_dep = fea_dep*m_glass
        fea_p3d = fea_p3d*m_edge
        res_low = self.cbam(self.out_conv(torch.cat((res_low, fea_dep), dim=1)))
        res_high = self.rcab(self.out_conv(torch.cat((res_high, fea_p3d), dim=1)))
        edge = self.sigmoid(self.pred_edge(res_high))
        out = res_low + res_high
        glass = self.sigmoid(self.pred_edge(out))
        # glass = self.sigmoid(self.pred_glass(res_low))
        # edge = self.sigmoid(self.pred_edge(res_high))
        return out, glass, edge


class Decoder(nn.Module):
    def __init__(self, channel_list=[128, 256, 512, 1024]):
        super(Decoder, self).__init__()
        self.upsample = UpGlass(channel_list[0], 64)
        self.db0 = Decoder_block(channel_list[0], 64)
        self.db1 = Decoder_block(channel_list[1], channel_list[0])
        self.db2 = Decoder_block(channel_list[2], channel_list[1])
        self.db3 = Decoder_block(channel_list[3], channel_list[2])

    def forward(self, fea_list, c_glass, c_edge, f_dep, f_p3d):
        f_d, glass3, edge3 = self.db3(fea_list[3], fea_list[2], f_dep, f_p3d, c_glass, c_edge)
        f_d, glass2, edge2 = self.db2(f_d, fea_list[1], f_dep, f_p3d, glass3, edge3)
        f_d, glass1, edge1 = self.db1(f_d, fea_list[0], f_dep, f_p3d, glass2, edge2)
        f_d_up = self.upsample(f_d)
        f_d, glass0, edge0 = self.db0(f_d, f_d_up, f_dep, f_p3d, glass1, edge1)
        return [glass0, glass1, glass2, glass3], [edge0, edge1, edge2, edge3]

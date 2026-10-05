import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
from skimage import morphology
import scipy.ndimage as ndimage

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H=None, W=None):
        if H is not None and W is not None:
            B, N, C = x.shape
            tx = x.transpose(1, 2).view(B, C, H, W)
            conv_x = self.dwconv(tx)
            return conv_x.flatten(2).transpose(1, 2)
        else:
            return self.dwconv(x)


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))


class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)

    def forward(self, x, H, W):
        feat = self.fc1(x)
        res = feat
        out = self.dwconv(feat, H, W)
        out = self.norm1(out + res)
        out = self.act(out)
        out = self.fc2(out)
        return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN_skip(dim, int(dim * 4))

    def forward(self, x, H, W):
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class MiT(nn.Module):
    def __init__(self, image_size, dims, layers, token_mlp_mode='mix_skip'):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 4, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(image_size // 4, patch_sizes[1], strides[1], padding_sizes[1],
                                                   dims[0], dims[1])
        self.patch_embed3 = OverlapPatchEmbeddings(image_size // 8, patch_sizes[2], strides[2], padding_sizes[2],
                                                   dims[1], dims[2])
        self.patch_embed4 = OverlapPatchEmbeddings(image_size // 16, patch_sizes[3], strides[3], padding_sizes[3],
                                                   dims[2], dims[3])

        self.block1 = nn.ModuleList(
            [TransformerBlock(dims[0], heads[0], reduction_ratios[0], token_mlp_mode) for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(dims[0])
        self.block2 = nn.ModuleList(
            [TransformerBlock(dims[1], heads[1], reduction_ratios[1], token_mlp_mode) for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(dims[1])
        self.block3 = nn.ModuleList(
            [TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp_mode) for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(dims[2])
        self.block4 = nn.ModuleList(
            [TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp_mode) for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(dims[3])

    def forward(self, x):
        B = x.shape[0]
        outs = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed2(x)
        for blk in self.block2: x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed3(x)
        for blk in self.block3: x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        x, H, W = self.patch_embed4(x)
        for blk in self.block4: x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs


class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if len(self.reduction_ratio) == 4:
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[0], reduction_ratio[0])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[1], reduction_ratio[1])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[2], reduction_ratio[2])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        if len(self.reduction_ratio) == 4:
            tem0 = x[:, :2304, :].reshape(B, 48, 48, C).permute(0, 3, 1, 2)
            tem1 = x[:, 2304:3456, :].reshape(B, 24, 24, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 3456:4176, :].reshape(B, 12, 12, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 4176:4464, :]
            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)
            return self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))
        return x


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)
        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x_atten)


class BridgeLayer_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head, reduction_ratios)
        self.norm2 = nn.LayerNorm(dims)
        self.mixffn1 = MixFFN_skip(dims, dims * 4)
        self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)

    def forward(self, inputs):
        B = inputs[0].shape[0]
        C = 64
        if isinstance(inputs, list):
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.attn(self.norm1(inputs))
        tx = self.norm2(tx1)

        tem1 = tx[:, :2304, :].reshape(B, -1, C)
        tem2 = tx[:, 2304:3456, :].reshape(B, -1, C * 2)
        tem3 = tx[:, 3456:4176, :].reshape(B, -1, C * 5)
        tem4 = tx[:, 4176:4464, :].reshape(B, -1, C * 8)

        m1f = self.mixffn1(tem1, 48, 48).reshape(B, -1, C)
        m2f = self.mixffn2(tem2, 24, 24).reshape(B, -1, C)
        m3f = self.mixffn3(tem3, 12, 12).reshape(B, -1, C)
        m4f = self.mixffn4(tem4, 6, 6).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        return tx1 + t1


class BridegeBlock_4(nn.Module):
    def __init__(self, dims, head, reduction_ratios):
        super().__init__()
        self.bridge_layer1 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer2 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer3 = BridgeLayer_4(dims, head, reduction_ratios)
        self.bridge_layer4 = BridgeLayer_4(dims, head, reduction_ratios)

    def forward(self, x):
        bridge1 = self.bridge_layer1(x)
        bridge2 = self.bridge_layer2(bridge1)
        bridge3 = self.bridge_layer3(bridge2)
        bridge4 = self.bridge_layer4(bridge3)
        B, _, C = bridge4.shape
        outs = []
        outs.append(bridge4[:, :2304, :].reshape(B, 48, 48, C).permute(0, 3, 1, 2))
        outs.append(bridge4[:, 2304:3456, :].reshape(B, 24, 24, C * 2).permute(0, 3, 1, 2))
        outs.append(bridge4[:, 3456:4176, :].reshape(B, 12, 12, C * 5).permute(0, 3, 1, 2))
        outs.append(bridge4[:, 4176:4464, :].reshape(B, 6, 6, C * 8).permute(0, 3, 1, 2))
        return outs


class LKA_Adapter(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, dropout=0.1, init_scale=0.001):
        super().__init__()
        self.dim = max(hidden_dim if hidden_dim else in_dim // 4, 8)
        self.adapter_down = nn.Linear(in_dim, self.dim)
        self.adapter_conv = nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=7, stride=1, padding=3,
                                      groups=self.dim)
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.adapter_up = nn.Linear(self.dim, in_dim)
        self.norm = nn.GroupNorm(1, in_dim)

        nn.init.zeros_(self.adapter_up.weight)
        if self.adapter_up.bias is not None:
            nn.init.zeros_(self.adapter_up.bias)
        self.gamma = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x):
        permuted = False
        H_orig, W_orig = None, None
        if x.dim() == 4 and x.shape[1] == self.norm.num_channels:
            H_orig, W_orig = x.shape[2], x.shape[3]
            residual = x.permute(0, 2, 3, 1)
            x_normed = self.norm(x).flatten(2).transpose(1, 2)
            permuted = True
        else:
            residual, x_normed = x, x

        B, N, C = x_normed.shape
        x_down = self.adapter_down(x_normed)
        H = H_orig if H_orig is not None else int(math.sqrt(N))
        W = W_orig if W_orig is not None else int(math.sqrt(N))

        x_spatial = x_down.transpose(1, 2).reshape(B, self.dim, H, W)
        x_spatial = self.adapter_conv(x_spatial)
        x_down = self.dropout(self.act(x_spatial.flatten(2).transpose(1, 2)))
        x_up = self.adapter_up(x_down)

        if permuted: x_up = x_up.reshape(B, H_orig, W_orig, C)
        out = residual + self.gamma * x_up
        return out.permute(0, 3, 1, 2).contiguous() if permuted else out


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        return self.norm(x.view(B, -1, C // 4).clone())


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        return self.norm(x.view(B, -1, self.output_dim).clone())


class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, heads, reduction_ratios, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims, out_dim = in_out_chan[0], in_out_chan[1]
        self.is_last = is_last
        if not is_last:
            self.concat_linear = nn.Linear(dims * 2, out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.concat_linear = nn.Linear(dims * 4, out_dim)
            self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4,
                                                norm_layer=norm_layer)
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)
        self.layer_former_2 = TransformerBlock(out_dim, heads, reduction_ratios, token_mlp_mode)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x1, x2=None):
        if x2 is not None:
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            cat_x = torch.cat([x1, x2], dim=-1)
            tran_layer_1 = self.layer_former_1(self.concat_linear(cat_x), h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)
            if self.last_layer:
                return self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            return self.layer_up(tran_layer_2)
        else:
            return self.layer_up(x1)


class GMMFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.norm_x = nn.LayerNorm(dim)
        self.norm_gmm = nn.LayerNorm(dim)

        self.align_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.gmm_gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.gmm_gate.weight)
        nn.init.zeros_(self.gmm_gate.bias)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x, gmm_features, uncertainty_mask=None, current_scale=0.1):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        x_norm = self.norm_x(x_flat)
        gmm_norm = self.norm_gmm(gmm_features)
        gmm_aligned = self.align_proj(gmm_norm)

        sa_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm)
        ca_out, _ = self.cross_attn(query=x_norm, key=gmm_aligned, value=gmm_aligned)

        combined_out = self.fusion_proj(torch.cat([sa_out, ca_out], dim=-1))

        if uncertainty_mask is not None:
            mask_resized = F.interpolate(uncertainty_mask, size=(H, W), mode='nearest')
            mask_flat = mask_resized.view(B, -1, 1)

            routed_attn = (1.0 - mask_flat) * sa_out + mask_flat * combined_out

            gmm_injection = self.gmm_gate(self.dropout(routed_attn)).view(B, H, W, C).permute(0, 3, 1, 2)
            fused = x + self.gamma.view(1, -1, 1, 1) * current_scale * gmm_injection
        else:
            gmm_injection = self.gmm_gate(self.dropout(combined_out)).view(B, H, W, C).permute(0, 3, 1, 2)
            fused = x + self.gamma.view(1, -1, 1, 1) * current_scale * gmm_injection

        return fused.contiguous(), ca_out


class MISSFormer(nn.Module):

    def __init__(self, num_classes=3, token_mlp_mode="mix_skip", encoder_pretrained=True,
                 apply_morph_postprocess=False):
        super().__init__()
        self.num_classes = num_classes
        self.apply_morph_postprocess = apply_morph_postprocess 

        self.aux_head = nn.Conv2d(512, num_classes, kernel_size=1)
        self.reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]
        d_base_feat_size = 6
        in_out_chan = [[32, 64], [144, 128], [288, 320], [512, 512]]
        dims, layers = [[64, 128, 320, 512], [2, 2, 2, 2]]

        self.backbone = MiT(192, dims, layers, token_mlp_mode)
        self.bridge = BridegeBlock_4(64, 1, [8, 4, 2, 1])

        self.gmm_fusions = nn.ModuleList([
            GMMFusionBlock(dim=dims[0], num_heads=2),
            GMMFusionBlock(dim=dims[1], num_heads=4),
            GMMFusionBlock(dim=dims[2], num_heads=8),
            GMMFusionBlock(dim=dims[3], num_heads=8)
        ])

        self.encoder_adapters = nn.ModuleList([
            nn.ModuleList([LKA_Adapter(64, 32), LKA_Adapter(128, 64), LKA_Adapter(320, 160), LKA_Adapter(512, 256)]),
            nn.ModuleList([LKA_Adapter(64, 32), LKA_Adapter(128, 64), LKA_Adapter(320, 160), LKA_Adapter(512, 256)]),
            nn.ModuleList([LKA_Adapter(64, 32), LKA_Adapter(128, 64), LKA_Adapter(320, 160), LKA_Adapter(512, 256)])
        ])

        self.decoder_adapters = nn.ModuleList([
            nn.ModuleList([LKA_Adapter(512, 256), LKA_Adapter(320, 160), LKA_Adapter(128, 64), LKA_Adapter(64, 32)]),
            nn.ModuleList([LKA_Adapter(512, 256), LKA_Adapter(320, 160), LKA_Adapter(128, 64), LKA_Adapter(64, 32)]),
            nn.ModuleList([LKA_Adapter(512, 256), LKA_Adapter(320, 160), LKA_Adapter(128, 64), LKA_Adapter(64, 32)])
        ])

        self.decoder_3 = MyDecoderLayer((d_base_feat_size, d_base_feat_size), in_out_chan[3], heads[3],
                                        self.reduction_ratios[3], token_mlp_mode, n_class=num_classes)
        self.decoder_2 = MyDecoderLayer((d_base_feat_size * 2, d_base_feat_size * 2), in_out_chan[2], heads[2],
                                        self.reduction_ratios[2], token_mlp_mode, n_class=num_classes)
        self.decoder_1 = MyDecoderLayer((d_base_feat_size * 4, d_base_feat_size * 4), in_out_chan[1], heads[1],
                                        self.reduction_ratios[1], token_mlp_mode, n_class=num_classes)
        self.decoder_0 = MyDecoderLayer((d_base_feat_size * 8, d_base_feat_size * 8), in_out_chan[0], heads[0],
                                        self.reduction_ratios[0], token_mlp_mode, n_class=num_classes, is_last=True)

        self.view_translators = nn.ModuleList([
            nn.Conv2d(dims[0], dims[0], 1),
            nn.Conv2d(dims[1], dims[1], 1),
            nn.Conv2d(dims[2], dims[2], 1),
            nn.Conv2d(dims[3], dims[3], 1)
        ])
        for m in self.view_translators:
            nn.init.dirac_(m.weight)
            nn.init.zeros_(m.bias)

    def _run_bridge_and_decoder(self, encoder_feats, direction):
        bridge = self.bridge(encoder_feats)
        b, c, _, _ = bridge[3].shape
        cur_dec_adapters = self.decoder_adapters[direction]

        feat_3 = cur_dec_adapters[0](bridge[3]).permute(0, 2, 3, 1).view(b, -1, c)
        tmp_3 = self.decoder_3(feat_3)

        feat_2 = cur_dec_adapters[1](bridge[2]).permute(0, 2, 3, 1)
        tmp_2 = self.decoder_2(tmp_3, feat_2)

        feat_1 = cur_dec_adapters[2](bridge[1]).permute(0, 2, 3, 1)
        tmp_1 = self.decoder_1(tmp_2, feat_1)

        feat_0 = cur_dec_adapters[3](bridge[0]).permute(0, 2, 3, 1)
        return self.decoder_0(tmp_1, feat_0)


    def forward(self, x, direction=0, gmm_samples=None, margin_threshold=0.5):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        encoder_outs = self.backbone(x)
        cur_enc_adapters = self.encoder_adapters[direction]
        base_enc_feats = [cur_enc_adapters[i](feat) for i, feat in enumerate(encoder_outs)]

        if gmm_samples is None:
            final_logits = self._run_bridge_and_decoder(base_enc_feats, direction)
            total_gmm_loss, aux_logits_up = None, None
        else:
            bridge_outs = self.bridge(base_enc_feats)
            aux_logits_raw = self.aux_head(bridge_outs[3])
            aux_logits_up = F.interpolate(aux_logits_raw, size=(192, 192), mode='bilinear', align_corners=False)

            aux_probs = torch.sigmoid(aux_logits_up)
            uncertainty = 1.0 - torch.abs(aux_probs - 0.5) * 2.0
            mean_uncertainty = uncertainty.mean(dim=1)
            uncertainty_mask = (mean_uncertainty > margin_threshold).float().unsqueeze(1)

            layer_scales = [0.01, 0.05, 0.15, 0.20]
            total_gmm_loss = torch.tensor(0.0, device=x.device)
            gmm_enc_feats = []

            for i, feat_adapter in enumerate(base_enc_feats):
                if len(gmm_samples) > i and gmm_samples[i] is not None:
                    current_gmm = gmm_samples[i]
                    B, C, H, W = feat_adapter.shape

                    if direction != 0:
                        feat_for_loss = self.view_translators[i](feat_adapter)
                    else:
                        feat_for_loss = feat_adapter

                    mask_resized = F.interpolate(uncertainty_mask, size=(H, W), mode='nearest')

                    feat_final, ca_gmm_out = self.gmm_fusions[i](
                        feat_for_loss, current_gmm,
                        uncertainty_mask=uncertainty_mask,
                        current_scale=layer_scales[i]
                    )

                    feat_flat = feat_for_loss.view(B, C, -1).permute(0, 2, 1)
                    feat_flat_norm = self.gmm_fusions[i].norm_x(feat_flat)
                    mask_flat = mask_resized.view(B, -1, 1)

                    diff = (feat_flat_norm - ca_gmm_out) ** 2
                    mask_sum = mask_flat.sum() * C + 1e-8
                    routing_loss = (diff * mask_flat).sum() / mask_sum

                    total_gmm_loss += routing_loss
                else:
                    feat_final = feat_adapter

                gmm_enc_feats.append(feat_final)

            final_logits = self._run_bridge_and_decoder(gmm_enc_feats, direction)
        if not self.training and self.apply_morph_postprocess:
            with torch.no_grad():

                probs = torch.softmax(final_logits, dim=1).cpu().numpy()[0]
                pred_3d = np.argmax(probs, axis=0).astype(np.uint8)


                tc_mask = (pred_3d == 1) | (pred_3d == 3)
                pred_3d[ndimage.binary_fill_holes(tc_mask) & (~tc_mask)] = 1

                et_mask = (pred_3d == 3)
                et_cleaned = morphology.remove_small_objects(et_mask, min_size=300)
                pred_3d[et_mask & (~et_cleaned)] = 1

                wt_mask = (pred_3d > 0)
                wt_cleaned = morphology.remove_small_objects(wt_mask, min_size=500)
                pred_3d[wt_mask & (~wt_cleaned)] = 0

                return pred_3d

        if gmm_samples is None:
            return final_logits
        return final_logits, total_gmm_loss, aux_logits_up

    def extract_class_aware_features(self, x, masks, direction=0):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        with torch.no_grad():
            encoder_outs = self.backbone(x)
            cur_adapters = self.encoder_adapters[direction]

            features_dict = {l: [] for l in range(len(encoder_outs))}
            for i, feat in enumerate(encoder_outs):
                feat_adapted = cur_adapters[i](feat)
                B, C, H, W = feat_adapted.shape
                flat_feat = feat_adapted.permute(0, 2, 3, 1).reshape(-1, C)
                features_dict[i].append(flat_feat.cpu().numpy())

        return features_dict
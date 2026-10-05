import argparse
import logging
import os
import random
import numpy as np
import torch
import joblib
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from networks.SDPT import MISSFormer
from datasets.dataset_brats19 import Brats19_dataset

import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
GREEN, RED, YELLOW, CYAN, END = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m'


class MoELayerGMM:
    def __init__(self, num_layers=4, num_experts=3, expert_components=5, pca_var=0.98, max_pca_dim=32):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_components = expert_components
        self.pca_var = pca_var
        self.max_pca_dim = max_pca_dim
        self.models = [{'scaler': None, 'pca': None, 'gmms': {}} for _ in range(num_layers)]

    def fit(self, features_by_layer):
        logging.info(f"\n{GREEN}>>> 开始训练 MoE-GMM (互斥真假双重过滤 / 动态PCA / Full协方差)...{END}")
        for l in range(self.num_layers):
            layer_feats = features_by_layer[l]
            valid_classes = [cls_id for cls_id, feats in layer_feats.items() if len(feats) > 0]
            if not valid_classes: continue

            is_deep = (l >= 2)
            cov_type = 'full'
            current_reg = 0.01 if is_deep else 0.005

            X_all = np.concatenate([layer_feats[cls_id] for cls_id in valid_classes], axis=0)
            scaler = RobustScaler()
            X_all_scaled = scaler.fit_transform(X_all)

            pca_temp = PCA(n_components=self.pca_var, whiten=False, random_state=42)
            pca_temp.fit(X_all_scaled)
            final_dim = max(min(pca_temp.n_components_, self.max_pca_dim), min(4, X_all_scaled.shape[1]))

            pca = PCA(n_components=final_dim, whiten=False, random_state=42)
            pca.fit(X_all_scaled)

            self.models[l]['scaler'] = scaler
            self.models[l]['pca'] = pca
            logging.info(
                f"   --- Layer {l} ({'深层' if is_deep else '浅层'}): PCA 自动降至 {pca.n_components_} 维 (保留 {self.pca_var * 100}% 信息)")

            for cls_id in valid_classes:
                X_cls = layer_feats[cls_id]
                if len(X_cls) < 100: continue
                X_cls_pca = pca.transform(scaler.transform(X_cls))

                router = GaussianMixture(n_components=self.num_experts, covariance_type=cov_type,
                                         random_state=42, reg_covar=current_reg, init_params='k-means++')
                router.fit(X_cls_pca)
                expert_assignments = router.predict(X_cls_pca)
                experts_dict = {}

                for expert_id in range(self.num_experts):
                    X_expert = X_cls_pca[expert_assignments == expert_id]
                    if len(X_expert) < 20: continue

                    n_comp = self.expert_components * 3 if is_deep else self.expert_components

                    # 🌟 修复 3: 动态安全锁。确保高斯核分配到的样本数足够支撑协方差矩阵计算
                    min_samples_per_comp = max(10, pca.n_components_ * 3)
                    actual_components = min(n_comp, max(1, len(X_expert) // min_samples_per_comp))

                    expert_gmm = GaussianMixture(n_components=actual_components,
                                                 covariance_type=cov_type,
                                                 random_state=42, reg_covar=current_reg, init_params='k-means++')
                    expert_gmm.fit(X_expert)
                    experts_dict[expert_id] = expert_gmm

                self.models[l]['gmms'][cls_id] = {'router': router, 'experts': experts_dict}
                logging.info(f"      ✅ 标签 '{cls_id}' 拟合成功.")

    def save(self, path):
        export_data = []
        for l in range(self.num_layers):
            layer_dict = {}
            if self.models[l]['scaler'] is None:
                export_data.append(None)
                continue
            for cls_id, moe_pack in self.models[l]['gmms'].items():
                layer_dict[cls_id] = {
                    'router': moe_pack['router'], 'experts': moe_pack['experts'],
                    'pca': self.models[l]['pca'], 'scaler': self.models[l]['scaler']
                }
            export_data.append(layer_dict)
        joblib.dump(export_data, path)
        logging.info(f"\n{RED}💾 伪标签专属字典已保存: {path}{END}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--list_dir', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--output_gmm_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--adapter_dir', type=int, default=2)
    parser.add_argument('--plant', type=str, default='sagittal')
    # parser.add_argument('--plant', type=str, default='coronal')
    parser.add_argument('--boundary_ratio', type=float, default=0.8)
    args = parser.parse_args()
    if not os.path.isfile(args.model_path):
        parser.error('--model_path must point to an existing directional checkpoint')
    if not os.path.isfile(os.path.join(args.list_dir, args.train_list)):
        parser.error('--train_list must identify an existing training split in --list_dir')
    if args.plant not in ('axial', 'coronal', 'sagittal') or args.adapter_dir != {'axial': 0, 'coronal': 1, 'sagittal': 2}.get(args.plant):
        parser.error('Use axial/0, coronal/1, or sagittal/2 for --plant/--adapter_dir')
    if not 0 <= args.boundary_ratio <= 1:
        parser.error('--boundary_ratio must be between 0 and 1')
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    net = MISSFormer(num_classes=3).to(device)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        net.load_state_dict(new_state_dict, strict=False)
        logging.info(f"{GREEN}✅ 成功加载 Phase 1 权重，准备自蒸馏！{END}")
    net.eval()

    features_by_layer = {}
    for l in range(4):
        # 🌟 修复 1: 浅层加入完整的 4 个互斥类 (0:BG, 1:Necrotic, 2:Edema, 3:Enhancing)
        if l < 2:
            features_by_layer[l] = {cls: {'boundary': [], 'core': []} for cls in [0, 1, 2, 3]}
        else:
            features_by_layer[l] = {cls: {'boundary': [], 'core': []} for cls in ['BG', 'FG']}

    feature_buffer = {}

    def get_hook(layer_idx):
        def hook(module, input, output):
            feature_buffer[layer_idx] = output.detach()

        return hook

    hooks = [net.encoder_adapters[args.adapter_dir][i].register_forward_hook(get_hook(i)) for i in range(4)]

    dataloader = DataLoader(
        Brats19_dataset(root=args.root_path, list_dir=args.list_dir, mode='train', plant=args.plant,
                        list_file_name=args.train_list),
        batch_size=8, shuffle=True, num_workers=2)

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(tqdm(dataloader, desc="特征与伪标签同步采集")):
            if i >= 100: break
            if imgs.dim() == 5:
                B, S, C, H, W = imgs.shape
                imgs = imgs.view(-1, C, H, W)
                masks = masks.view(-1, H, W)

            masks = masks.to(device)
            masks[masks == 4] = 3

            feature_buffer.clear()
            outputs = net(imgs.to(device), direction=args.adapter_dir)

            final_logits = outputs[0] if isinstance(outputs, tuple) else outputs
            final_preds = (torch.sigmoid(final_logits) > 0.5).float()

            for layer_idx in range(4):
                feat = feature_buffer[layer_idx]

                layer_preds = F.adaptive_avg_pool2d(final_preds, output_size=(feat.shape[2], feat.shape[3]))
                layer_preds_bin = (layer_preds > 0.5).float()

                masks_onehot = F.one_hot(masks.long(), num_classes=4).permute(0, 3, 1, 2).float()
                layer_true_prob = F.adaptive_avg_pool2d(masks_onehot, output_size=(feat.shape[2], feat.shape[3]))
                layer_true_bin = torch.argmax(layer_true_prob, dim=1)

                feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])

                # 🌟 修复 2: 嵌套标签转为绝对互斥标签
                pred_WT = layer_preds_bin[:, 0]
                pred_TC = layer_preds_bin[:, 1]
                pred_ET = layer_preds_bin[:, 2]

                pred_exclusive = torch.zeros_like(pred_WT)
                pred_exclusive[pred_WT == 1] = 2  # 水肿
                pred_exclusive[pred_TC == 1] = 1  # 坏死
                pred_exclusive[pred_ET == 1] = 3  # 增强

                # 动态分配 Target Masks (True Positive 采集)
                target_masks = {}
                if layer_idx < 2:
                    target_masks[0] = ((pred_exclusive == 0) & (layer_true_bin == 0)).float()
                    target_masks[1] = ((pred_exclusive == 1) & (layer_true_bin == 1)).float()
                    target_masks[2] = ((pred_exclusive == 2) & (layer_true_bin == 2)).float()
                    target_masks[3] = ((pred_exclusive == 3) & (layer_true_bin == 3)).float()
                else:
                    is_pred_fg = (pred_WT == 1).float()
                    is_true_fg = (layer_true_bin > 0).float()
                    target_masks['BG'] = ((1.0 - is_pred_fg) * (1.0 - is_true_fg)).float()
                    target_masks['FG'] = (is_pred_fg * is_true_fg).float()

                for cls_id, b_m in target_masks.items():
                    if b_m.sum() == 0: continue
                    eroded = -F.max_pool2d(-b_m.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
                    is_cls, is_core = (b_m == 1).view(-1), (eroded == 1).view(-1)
                    b_idx = (is_cls & (~is_core)).nonzero(as_tuple=True)[0]
                    c_idx = (is_cls & is_core).nonzero(as_tuple=True)[0]

                    if len(b_idx) > 0:
                        features_by_layer[layer_idx][cls_id]['boundary'].append(
                            feat_flat[b_idx[torch.randperm(len(b_idx))[:500]]].detach().cpu().numpy())
                    if len(c_idx) > 0:
                        features_by_layer[layer_idx][cls_id]['core'].append(
                            feat_flat[c_idx[torch.randperm(len(c_idx))[:500]]].detach().cpu().numpy())

    for h in hooks: h.remove()

    final_features = {l: {} for l in range(4)}
    for l in range(4):
        # 🌟 修复 1 附属修改: 拼装时照顾全部 4 个类
        keys = [0, 1, 2, 3] if l < 2 else ['BG', 'FG']
        for cls in keys:
            b_list = features_by_layer[l][cls]['boundary']
            c_list = features_by_layer[l][cls]['core']

            b_feats = np.concatenate(b_list, axis=0) if b_list else None
            c_feats = np.concatenate(c_list, axis=0) if c_list else None

            len_b = len(b_feats) if b_feats is not None else 0
            len_c = len(c_feats) if c_feats is not None else 0

            if len_b + len_c < 500:
                continue

            n_samples = 15000
            n_b = min(len_b, int(n_samples * args.boundary_ratio))
            n_c = min(len_c, n_samples - n_b)

            to_concat = []
            if n_b > 0:
                to_concat.append(b_feats[np.random.choice(len_b, n_b, replace=False)])
            if n_c > 0:
                to_concat.append(c_feats[np.random.choice(len_c, n_c, replace=False)])

            if to_concat:
                combined = np.concatenate(to_concat, axis=0)
                np.random.shuffle(combined)
                final_features[l][cls] = combined

    gmm_manager = MoELayerGMM()
    gmm_manager.fit(final_features)

    # 确保保存路径存在
    os.makedirs(args.output_gmm_dir, exist_ok=True)
    gmm_manager.save(os.path.join(args.output_gmm_dir, f'LKA_GMM_Final_{args.boundary_ratio}_{args.plant}.pkl'))


if __name__ == "__main__":
    main()

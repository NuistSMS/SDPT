import argparse
import logging
import os
import random
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from medpy import metric
import torch.backends.cudnn as cudnn
import yaml


from networks.SDPT import MISSFormer
from datasets.dataset_brats19 import Brats19_dataset

import warnings

warnings.filterwarnings('ignore')


GREEN, CYAN, YELLOW, RED, END = '\033[92m', '\033[96m', '\033[93m', '\033[91m', '\033[0m'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"{CYAN}Random seed: {seed} (cuDNN deterministic mode enabled){END}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calculate_dice(pred, gt):
    def binary_dice(p, g):
        p = p.contiguous().view(-1).cpu().numpy()
        g = g.contiguous().view(-1).cpu().numpy()
        if g.sum() == 0:
            return 1.0 if p.sum() == 0 else 0.0
        return metric.binary.dc(p, g)

    return binary_dice(pred[:, 0], gt[:, 0]), binary_dice(pred[:, 1], gt[:, 1]), binary_dice(pred[:, 2], gt[:, 2])



class CachedGMMSampler:
    def __init__(self, gmm_models_list, device, pool_size=100000, name="GMM"):
        self.device = device
        self.pool_size = pool_size
        self.feature_pools = {}
        print(f"{CYAN}[{name}] Pre-generating the feature pool...{END}")

        for l, layer_models in enumerate(gmm_models_list):
            if not layer_models: continue
            self.feature_pools[l] = {}
            for cls_id, moe in layer_models.items():
                all_weights, all_means, all_covs = [], [], []
                cov_type = 'full'

                for exp_id, expert_gmm in moe['experts'].items():
                    r_weight = moe['router'].weights_[exp_id]
                    for comp_idx in range(expert_gmm.n_components):
                        all_weights.append(r_weight * expert_gmm.weights_[comp_idx])
                        all_means.append(expert_gmm.means_[comp_idx])
                        all_covs.append(expert_gmm.covariances_[comp_idx])
                    cov_type = expert_gmm.covariance_type

                weights = torch.tensor(all_weights, dtype=torch.float32)
                weights /= weights.sum()
                means = torch.tensor(np.stack(all_means), dtype=torch.float32)
                covs = torch.tensor(np.stack(all_covs), dtype=torch.float32)

                idx = torch.multinomial(weights, pool_size, replacement=True)
                z = torch.zeros((pool_size, means.shape[1]), dtype=torch.float32)

                for comp_idx in range(len(weights)):
                    comp_mask = (idx == comp_idx)
                    num_samples_for_comp = comp_mask.sum().item()
                    if num_samples_for_comp == 0: continue

                    if cov_type == 'full':
                        safe_cov = covs[comp_idx] + torch.eye(covs[comp_idx].shape[0]) * 1e-4
                        dist = torch.distributions.MultivariateNormal(means[comp_idx], covariance_matrix=safe_cov)
                        z[comp_mask] = dist.sample((num_samples_for_comp,))
                    else:
                        dist = torch.distributions.Normal(means[comp_idx], torch.sqrt(covs[comp_idx] + 1e-6))
                        z[comp_mask] = dist.sample((num_samples_for_comp,))


                pca_comp = torch.tensor(moe['pca'].components_, dtype=torch.float32)
                pca_mean = torch.tensor(moe['pca'].mean_, dtype=torch.float32)
                features = torch.matmul(z, pca_comp) + pca_mean


                if moe['scaler'] is not None:
                    scale = torch.tensor(moe['scaler'].scale_, dtype=torch.float32)
                    shift = torch.tensor(
                        moe['scaler'].center_ if hasattr(moe['scaler'], 'center_') else getattr(moe['scaler'], 'mean_',
                                                                                                0),
                        dtype=torch.float32
                    )
                    features = features * scale + shift

                self.feature_pools[l][cls_id] = features.to(device)

        print(f"{GREEN}[{name}] Feature pool is ready.{END}")

    def sample_features(self, l, B, sl, cls_list):
        if l not in self.feature_pools: return None


        active_pools = [self.feature_pools[l][c] for c in cls_list if c in self.feature_pools[l]]
        if not active_pools:
            return None

        combined_pool = torch.cat(active_pools, dim=0)
        total_need = B * sl

        rand_indices = torch.randperm(len(combined_pool))[:total_need]
        return combined_pool[rand_indices].view(B, sl, -1)


class DualGMMSampler:
    def __init__(self, sampler_ax, sampler_cor, mode='random'):
        self.sampler_ax = sampler_ax
        self.sampler_cor = sampler_cor
        self.mode = mode

    def sample_features_all_layers(self, B, sls, cls_list):

        use_ax = True
        if self.mode == 'random':
            use_ax = random.random() < 0.5

        gmm_l = []
        for l in range(4):
            current_cls_list = cls_list if l < 2 else ['BG', 'FG']

            if self.mode == 'mix':
                f_ax = self.sampler_ax.sample_features(l, B, sls[l], current_cls_list)
                f_cor = self.sampler_cor.sample_features(l, B, sls[l], current_cls_list)
                gmm_l.append((f_ax + f_cor) / 2.0 if (f_ax is not None and f_cor is not None) else None)
            else:
                sampler = self.sampler_ax if use_ax else self.sampler_cor
                gmm_l.append(sampler.sample_features(l, B, sls[l], current_cls_list))
        return gmm_l



class RegionDecoupledLossPhase1(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(RegionDecoupledLossPhase1, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w, self.dice_w = bce_weight, dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_scores = (2. * intersection + smooth) / (union + smooth)
        weighted_dice = (dice_scores[:, 0] + dice_scores[:, 1] + dice_scores[:, 2]) / 3.0
        return self.bce_w * bce_loss + self.dice_w * (1.0 - weighted_dice.mean())


def train(trainloader, model, loss_obj, optimizer, current_dir, args, gmm_sampler):
    model.train()

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            module.eval()

    metrics = {"total": [], "seg": [], "aux": [], "gmm": [], "mask_ratio": [], "WT": [], "TC": [], "ET": []}
    iterator = tqdm(total=len(trainloader), ncols=180, desc="Training")
    optimizer.zero_grad()

    for i_vol, sampled_batch in enumerate(trainloader):
        full_imgs = sampled_batch[0].squeeze(0).cuda(non_blocking=True)
        full_labels = sampled_batch[1].squeeze(0).squeeze(1).cuda(non_blocking=True)
        full_labels[full_labels == 4] = 3

        num_slices = full_imgs.shape[0]
        step_size = args.train_step_size


        v_tot, v_seg, v_aux, v_gmm, v_mask, v_wt, v_tc, v_et, num_steps = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

        for i in range(0, num_slices, step_size):
            img_chunk = full_imgs[i: i + step_size]
            lbl_chunk = full_labels[i: i + step_size]
            B = img_chunk.shape[0]

            gmm_l = None
            if gmm_sampler is not None:
                p_cls = [0, 1, 2, 3]

                sls = [1024, 1024, 1024, 1024]
                gmm_l = gmm_sampler.sample_features_all_layers(B, sls, p_cls)


            outputs = model(img_chunk, direction=current_dir, gmm_samples=gmm_l, margin_threshold=args.margin_threshold)


            logits, gmm_loss, aux_logits = outputs[0], outputs[1], outputs[2]

            targets = torch.stack(
                [(lbl_chunk > 0).float(), ((lbl_chunk == 1) | (lbl_chunk == 3)).float(), (lbl_chunk == 3).float()],
                dim=1)


            seg_loss = loss_obj(logits, targets)
            aux_loss = loss_obj(aux_logits, targets)


            total_loss = (seg_loss + 0.4 * aux_loss + args.gmm_weight * gmm_loss) / (num_slices // step_size + 1)
            total_loss.backward()

            with torch.no_grad():

                aux_probs = torch.sigmoid(aux_logits)
                uncertainty = 1.0 - torch.abs(aux_probs - 0.5) * 2.0
                mean_uncertainty = uncertainty.mean(dim=1)
                active_ratio = (mean_uncertainty > args.margin_threshold).float().mean().item()
                v_mask += active_ratio


                preds = (torch.sigmoid(logits) > 0.5).float()
                wt, tc, et = calculate_dice(preds, targets)
                v_wt += wt
                v_tc += tc
                v_et += et

            v_tot += total_loss.item() * (num_slices // step_size + 1)
            v_seg += seg_loss.item()
            v_aux += aux_loss.item()
            v_gmm += gmm_loss.item()
            num_steps += 1

        optimizer.step()
        optimizer.zero_grad()


        metrics["total"].append(v_tot / num_steps)
        metrics["seg"].append(v_seg / num_steps)
        metrics["aux"].append(v_aux / num_steps)
        metrics["gmm"].append(v_gmm / num_steps)
        metrics["mask_ratio"].append(v_mask / num_steps)
        metrics["WT"].append(v_wt / num_steps)
        metrics["TC"].append(v_tc / num_steps)
        metrics["ET"].append(v_et / num_steps)

        iterator.update(1)

        iterator.set_postfix({
            'Dice': f"{(np.mean(metrics['WT']) + np.mean(metrics['TC']) + np.mean(metrics['ET'])) / 3:.3f}",
            'Tot': f"{np.mean(metrics['total']):.3f}",
            'Seg': f"{np.mean(metrics['seg']):.3f}",
            'Aux': f"{np.mean(metrics['aux']):.3f}",
            'GMM': f"{np.mean(metrics['gmm']):.4f}",
            'Act': f"{np.mean(metrics['mask_ratio']) * 100:.3f}%"
        })

    iterator.close()
    return np.mean(metrics["total"]), np.mean(metrics["seg"]), np.mean(metrics["aux"]), np.mean(
        metrics["gmm"]), np.mean(metrics["WT"]), np.mean(metrics["TC"]), np.mean(metrics["ET"])


def validate(valloader, model, current_dir):
    model.eval()
    v_wt, v_tc, v_et = [], [], []
    with torch.no_grad():
        for sampled_batch in tqdm(valloader, desc="Validating", ncols=120):
            imgs = sampled_batch[0].squeeze(0).cuda()
            lbls = sampled_batch[1].squeeze(0).squeeze(1).cuda();
            lbls[lbls == 4] = 3

            patient_preds, patient_targs = [], []
            for i in range(0, imgs.shape[0], 8):
                img_c = imgs[i:i + 8]
                lbl_c = lbls[i:i + 8]
                out = model(img_c, direction=current_dir)
                logits = out[0] if isinstance(out, tuple) else out
                preds = (torch.sigmoid(logits) > 0.5).float()
                targs = torch.stack([(lbl_c > 0).float(), ((lbl_c == 1) | (lbl_c == 3)).float(), (lbl_c == 3).float()],
                                    dim=1)
                patient_preds.append(preds.cpu());
                patient_targs.append(targs.cpu())

            full_preds, full_targs = torch.cat(patient_preds, 0).numpy(), torch.cat(patient_targs, 0).numpy()

            def get_3d_dice(p, t):
                return metric.binary.dc(p, t) if t.sum() > 0 else (1.0 if p.sum() == 0 else 0.0)

            v_wt.append(get_3d_dice(full_preds[:, 0], full_targs[:, 0]))
            v_tc.append(get_3d_dice(full_preds[:, 1], full_targs[:, 1]))
            v_et.append(get_3d_dice(full_preds[:, 2], full_targs[:, 2]))
    return np.mean(v_wt), np.mean(v_tc), np.mean(v_et)


def trainer_brats19(args, model, current_dir=2):
    cudnn.benchmark = False
    cudnn.deterministic = True

    db_train = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, root=args.root_path, mode='train',
                               list_file_name=args.train_list)
    db_val = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, root=args.root_path, mode='valid',
                              list_file_name=args.val_list)
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                              worker_init_fn=seed_worker)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    required_paths = (
        args.resume_ckpt, args.gmm_path_ax_05, args.gmm_path_ax_08,
        args.gmm_path_cor_05, args.gmm_path_cor_08,
    )
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required input not found: {path}")

    ax_sampler_05 = CachedGMMSampler(joblib.load(args.gmm_path_ax_05), device=torch.device('cuda:0'), name="Axial 0.5")
    ax_sampler_08 = CachedGMMSampler(joblib.load(args.gmm_path_ax_08), device=torch.device('cuda:0'), name="Axial 0.8")
    cor_sampler_05 = CachedGMMSampler(joblib.load(args.gmm_path_cor_05), device=torch.device('cuda:0'),
                                      name="Coronal 0.5")
    cor_sampler_08 = CachedGMMSampler(joblib.load(args.gmm_path_cor_08), device=torch.device('cuda:0'),
                                      name="Coronal 0.8")

    dual_sampler_05 = DualGMMSampler(ax_sampler_05, cor_sampler_05, mode='random')
    dual_sampler_08 = DualGMMSampler(ax_sampler_08, cor_sampler_08, mode='random')


    model_module = model.module if hasattr(model, 'module') else model
    checkpoint = torch.load(args.resume_ckpt, map_location='cuda:0', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    curr_dict = model_module.state_dict()
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()
                  if k.replace('module.', '') in curr_dict and v.shape == curr_dict[k.replace('module.', '')].shape}
    model_module.load_state_dict(clean_dict, strict=False)


    for m in model_module.view_translators:
        nn.init.dirac_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)


    for param in model.parameters(): param.requires_grad = False
    for name, param in model_module.named_parameters():
        if any(kw in name for kw in
               [f"adapters.{current_dir}.", "last_layer", "aux_head", "decoder_0", "decoder_1", "gmm_fusions",
                "view_translators"]):
            param.requires_grad = True

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    loss_obj = RegionDecoupledLossPhase1()

    top_k_checkpoints = []
    K = 15
    for epoch in range(args.max_epochs):

        curr_dual_sampler = dual_sampler_05 if epoch < args.switch_epoch else dual_sampler_08
        print(
            f"\nEpoch {epoch} | Dual guide (random selection) | Ratio: [{'0.5' if epoch < args.switch_epoch else '0.8'}]")


        t_tot, t_seg, t_aux, t_gmm, t_wt, t_tc, t_et = train(trainloader, model, loss_obj, optimizer, current_dir, args,
                                                             curr_dual_sampler)


        print(
            f"{YELLOW}[Train loss] Total: {t_tot:.4f} | Seg: {t_seg:.4f} | Aux: {t_aux:.4f} | GMM: {t_gmm:.4f}{END}")

        v_wt, v_tc, v_et = validate(valloader, model, current_dir)
        v_avg = (v_wt + v_tc + v_et) / 3.0
        print(f"{CYAN}[Validation] Avg: {v_avg:.4f} | WT: {v_wt:.4f} | TC: {v_tc:.4f} | ET: {v_et:.4f}{END}")

        scheduler.step()
        is_top_k = False

        if len(top_k_checkpoints) < K:
            is_top_k = True
        else:
            top_k_checkpoints.sort(key=lambda x: x[0])
            lowest_top_k_score = top_k_checkpoints[0][0]
            if v_avg > lowest_top_k_score:
                is_top_k = True
                _, file_to_delete = top_k_checkpoints.pop(0)
                if os.path.exists(file_to_delete):
                    os.remove(file_to_delete)

        if is_top_k:
            save_path = os.path.join(args.output_dir, f"Top{K}_Epoch{epoch}_Avg{v_avg:.4f}_DIR{current_dir}.pth")
            torch.save({'model_state_dict': model_module.state_dict(), 'dice': v_avg, 'epoch': epoch}, save_path)
            top_k_checkpoints.append((v_avg, save_path))
            current_min_score = min([x[0] for x in top_k_checkpoints])
            print(
                f"{GREEN}Saved to the top-{K} checkpoints: {os.path.basename(save_path)} (cutoff: {current_min_score:.4f}){END}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: sagittal SDPT training with axial and coronal priors")
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--list_dir', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--val_list', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--base_lr', type=float, default=2e-4)

    parser.add_argument('--plant', type=str, default='sagittal', choices=['sagittal'])
    parser.add_argument('--train_step_size', type=int, default=8)
    parser.add_argument('--resume_ckpt', type=str, required=True)
    parser.add_argument('--switch_epoch', type=int, default=6)

    parser.add_argument('--gmm_path_ax_05', type=str, required=True)
    parser.add_argument('--gmm_path_ax_08', type=str, required=True)
    parser.add_argument('--gmm_path_cor_05', type=str, required=True)
    parser.add_argument('--gmm_path_cor_08', type=str, required=True)

    parser.add_argument('--gmm_weight', type=float, default=0.05)
    parser.add_argument('--margin_threshold', type=float, default=0.65)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    setup_seed(args.seed)

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'train_config.yaml'), 'w', encoding='utf-8') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False, allow_unicode=True)
    net = MISSFormer(num_classes=3).cuda(0)

    trainer_brats19(args, net, current_dir=2)

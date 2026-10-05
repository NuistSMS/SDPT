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
        if g.sum() == 0: return 1.0 if p.sum() == 0 else 0.0
        return metric.binary.dc(p, g)

    dice_wt = binary_dice(pred[:, 0], gt[:, 0])
    dice_tc = binary_dice(pred[:, 1], gt[:, 1])
    dice_et = binary_dice(pred[:, 2], gt[:, 2])
    return dice_wt, dice_tc, dice_et


class CachedGMMSampler:
    def __init__(self, gmm_models_list, device, pool_size=150000):
        self.device = device
        self.pool_size = pool_size
        self.feature_pools = {}
        print(f"{CYAN}Loading the decoupled GMM bank (L0-1: three regions; L2-3: foreground/background)...{END}")

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
                    if comp_mask.sum() == 0: continue
                    if cov_type == 'full':
                        safe_cov = covs[comp_idx] + torch.eye(covs[comp_idx].shape[0]) * 1e-4
                        dist = torch.distributions.MultivariateNormal(means[comp_idx], covariance_matrix=safe_cov)
                        z[comp_mask] = dist.sample((comp_mask.sum().item(),))
                    else:
                        dist = torch.distributions.Normal(means[comp_idx], torch.sqrt(covs[comp_idx] + 1e-6))
                        z[comp_mask] = dist.sample((comp_mask.sum().item(),))

                pca_comp = torch.tensor(moe['pca'].components_, dtype=torch.float32)
                pca_mean = torch.tensor(moe['pca'].mean_, dtype=torch.float32)
                features = torch.matmul(z, pca_comp) + pca_mean
                if moe['scaler'] is not None:
                    scale = torch.tensor(moe['scaler'].scale_, dtype=torch.float32)
                    shift = torch.tensor(
                        moe['scaler'].center_ if hasattr(moe['scaler'], 'center_') else getattr(moe['scaler'], 'mean_',
                                                                                                0), dtype=torch.float32)
                    features = features * scale + shift
                self.feature_pools[l][cls_id] = features.to(device)
        print(f"{GREEN}GMM sampling pools are ready.{END}")

    def sample_features(self, l, B, sl, cls_list):
        if l not in self.feature_pools: return None
        available_cls = [c for c in cls_list if c in self.feature_pools[l]]
        if not available_cls:
            target_c = list(self.feature_pools[l].keys())[0]
        else:
            target_c = random.choice(available_cls)

        pool = self.feature_pools[l][target_c]
        total_need = B * sl
        start_idx = random.randint(0, self.pool_size - total_need - 1)
        return pool[start_idx: start_idx + total_need].view(B, sl, -1)


class RegionDecoupledLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(RegionDecoupledLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w, self.dice_w = bce_weight, dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_scores = (2. * intersection + 1e-5) / (union + 1e-5)
        return self.bce_w * bce_loss + self.dice_w * (1.0 - dice_scores.mean())


def train(trainloader, model, loss_obj, optimizer, current_dir, args, gmm_sampler):
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            module.eval()


    metrics = {"total": [], "seg": [], "aux": [], "gmm": [], "mask_ratio": [], "WT": [], "TC": [], "ET": []}
    iterator = tqdm(total=len(trainloader), ncols=180, desc="GMM-guided training")
    optimizer.zero_grad()

    for i_vol, sampled_batch in enumerate(trainloader):
        imgs = sampled_batch[0].squeeze(0).cuda(non_blocking=True)
        labels = sampled_batch[1].squeeze(0).squeeze(1).cuda(non_blocking=True)
        labels[labels == 4] = 3
        num_slices = imgs.shape[0]

        v_tot, v_seg, v_aux, v_gmm, v_mask, v_wt, v_tc, v_et, n_s = 0, 0, 0, 0, 0, 0, 0, 0, 0

        for i in range(0, num_slices, args.train_step_size):
            img_c = imgs[i: i + args.train_step_size]
            lbl_c = labels[i: i + args.train_step_size]
            B = img_c.shape[0]

            gmm_l = None
            if gmm_sampler is not None:
                gmm_l = []
                for l in range(4):
                    cls_pool = [0, 1, 2, 3] if l < 2 else ['BG', 'FG']
                    sl = (img_c.shape[2] // (4 * (2 ** l))) * (img_c.shape[3] // (4 * (2 ** l)))
                    gmm_l.append(gmm_sampler.sample_features(l, B, sl, cls_pool))

            outputs = model(img_c, direction=current_dir, gmm_samples=gmm_l, margin_threshold=args.margin_threshold)
            logits, gmm_loss, aux_logits = outputs[0], outputs[1], outputs[2]

            targets = torch.stack([(lbl_c > 0).float(), ((lbl_c == 1) | (lbl_c == 3)).float(), (lbl_c == 3).float()],
                                  dim=1)


            seg_loss = loss_obj(logits, targets)
            aux_loss = loss_obj(aux_logits, targets)

            total_loss = (seg_loss + 0.4 * aux_loss + args.gmm_weight * gmm_loss) / (
                        num_slices // args.train_step_size + 1)
            total_loss.backward()

            with torch.no_grad():

                aux_probs = torch.sigmoid(aux_logits)
                uncertainty = 1.0 - torch.abs(aux_probs - 0.5) * 2.0
                mean_uncertainty = uncertainty.mean(dim=1)
                active_ratio = (mean_uncertainty > args.margin_threshold).float().mean().item()
                v_mask += active_ratio


                preds = (torch.sigmoid(logits) > 0.5).float()
                wt, tc, et = calculate_dice(preds, targets)
                v_wt += wt;
                v_tc += tc;
                v_et += et


            v_tot += total_loss.item() * (num_slices // args.train_step_size + 1)
            v_seg += seg_loss.item()
            v_aux += aux_loss.item()
            v_gmm += gmm_loss.item()
            n_s += 1

        optimizer.step()
        optimizer.zero_grad()


        metrics["total"].append(v_tot / n_s)
        metrics["seg"].append(v_seg / n_s)
        metrics["aux"].append(v_aux / n_s)
        metrics["gmm"].append(v_gmm / n_s)
        metrics["mask_ratio"].append(v_mask / n_s)
        metrics["WT"].append(v_wt / n_s)
        metrics["TC"].append(v_tc / n_s)
        metrics["ET"].append(v_et / n_s)

        iterator.update(1)

        iterator.set_postfix({
            'Dice': f"{(np.mean(metrics['WT']) + np.mean(metrics['TC']) + np.mean(metrics['ET'])) / 3:.3f}",
            'Tot': f"{np.mean(metrics['total']):.3f}",
            'Seg': f"{np.mean(metrics['seg']):.3f}",
            'Aux': f"{np.mean(metrics['aux']):.3f}",
            'GMM': f"{np.mean(metrics['gmm']):.4f}",
            'Act': f"{np.mean(metrics['mask_ratio']) * 100:.1f}%"
        })

    iterator.close()

    return np.mean(metrics["total"]), np.mean(metrics["seg"]), np.mean(metrics["aux"]), np.mean(
        metrics["gmm"]), np.mean(metrics["WT"]), np.mean(metrics["TC"]), np.mean(metrics["ET"])


def validate(valloader, model, current_dir):
    model.eval()
    v_wt, v_tc, v_et = [], [], []
    with torch.no_grad():
        for sampled_batch in tqdm(valloader, desc="Validating"):
            imgs = sampled_batch[0].squeeze(0).cuda()
            lbls = sampled_batch[1].squeeze(0).squeeze(1).cuda()
            lbls[lbls == 4] = 3
            p_preds, p_targs = [], []

            for i in range(0, imgs.shape[0], 8):
                out = model(imgs[i:i + 8], direction=current_dir)

                p_preds.append((torch.sigmoid(out) > 0.5).float().cpu())
                p_targs.append(torch.stack(
                    [(lbls[i:i + 8] > 0).float(), ((lbls[i:i + 8] == 1) | (lbls[i:i + 8] == 3)).float(),
                     (lbls[i:i + 8] == 3).float()], dim=1).cpu())

            f_p, f_t = torch.cat(p_preds, 0).numpy(), torch.cat(p_targs, 0).numpy()
            v_wt.append(metric.binary.dc(f_p[:, 0], f_t[:, 0]) if f_t[:, 0].sum() > 0 else 1.0)
            v_tc.append(metric.binary.dc(f_p[:, 1], f_t[:, 1]) if f_t[:, 1].sum() > 0 else 1.0)
            v_et.append(metric.binary.dc(f_p[:, 2], f_t[:, 2]) if f_t[:, 2].sum() > 0 else 1.0)
    return np.mean(v_wt), np.mean(v_tc), np.mean(v_et)


def trainer_brats19(args, model, current_dir=2):
    db_train = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, root=args.root_path, mode='train',
                               list_file_name=args.train_list)
    db_val = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, root=args.root_path, mode='valid',
                              list_file_name=args.val_list)
    trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=args.num_workers,
                             worker_init_fn=seed_worker, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)

    for path in (args.gmm_path_05, args.gmm_path_08, args.resume_ckpt):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required input not found: {path}")
    sampler_05 = CachedGMMSampler(joblib.load(args.gmm_path_05), device=torch.device('cuda:0'))
    sampler_08 = CachedGMMSampler(joblib.load(args.gmm_path_08), device=torch.device('cuda:0'))

    model_module = model.module if hasattr(model, 'module') else model
    checkpoint = torch.load(args.resume_ckpt, map_location='cuda:0', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model_module.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items() if
                                  k.replace('module.', '') in model_module.state_dict() and 'last_layer' not in k},
                                 strict=False)

    for param in model.parameters(): param.requires_grad = False
    for name, param in model_module.named_parameters():
        if any(kw in name for kw in
               [f"adapters.{current_dir}.", "last_layer", "aux_head", "decoder_0", "decoder_1", "gmm_fusions",
                "view_translators"]):
            param.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_dict = {
        "1. Backbone": 0,
        "2. Adapters": 0,
        "3. New decoders": 0,
        "4. GMM fusion layers": 0,
        "5. View translators": 0,
        "6. Segmentation head": 0,
        "7. Auxiliary head": 0,
        "8. Other": 0
    }

    for name, param in model_module.named_parameters():
        num = param.numel()
        if "adapters" in name:
            param_dict["2. Adapters"] += num
        elif "decoder_0" in name or "decoder_1" in name:
            param_dict["3. New decoders"] += num
        elif "gmm_fusions" in name:
            param_dict["4. GMM fusion layers"] += num
        elif "view_translators" in name:
            param_dict["5. View translators"] += num
        elif "last_layer" in name:
            param_dict["6. Segmentation head"] += num
        elif "aux_head" in name:
            param_dict["7. Auxiliary head"] += num
        elif any(kw in name for kw in ["encoder", "patch_embed", "pos_embed", "decoder"]):  
            param_dict["1. Backbone"] += num
        else:
            param_dict["8. Other"] += num

    print(f"\n{CYAN}================= Model parameter summary ================={END}")
    for k, v in param_dict.items():
        if v > 0:
            print(f"{CYAN}   - {k:<35}: {v / 1e6:>6.2f} M{END}")
    print(f"{CYAN}   ----------------------------------------------------{END}")
    print(f"{GREEN}   - Total parameters                    : {total_params / 1e6:>6.2f} M{END}")
    print(
        f"{YELLOW}   - Trainable parameters                : {trainable_params / 1e6:>6.2f} M ({(trainable_params / total_params) * 100:.1f}%){END}")
    print(f"{CYAN}======================================================={END}\n")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    loss_obj = RegionDecoupledLoss()

    top_k_checkpoints = []
    K = 15
    for epoch in range(args.max_epochs):
        curr_sampler = sampler_05 if epoch < args.switch_epoch else sampler_08
        print(f"\nEpoch {epoch} | Global top-2 mode")


        t_tot, t_seg, t_aux, t_gmm, t_wt, t_tc, t_et = train(trainloader, model, loss_obj, optimizer, current_dir, args,
                                                             curr_sampler)


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
    parser = argparse.ArgumentParser(description="Stage 2: train one SDPT direction with a completed-view GMM prior")
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--list_dir', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--val_list', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--base_lr', type=float, default=2e-4)
    parser.add_argument('--plant', type=str, default='coronal', choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--train_step_size', type=int, default=8)
    parser.add_argument('--resume_ckpt', type=str, required=True)
    parser.add_argument('--switch_epoch', type=int, default=60)
    parser.add_argument('--gmm_path_05', type=str, required=True)
    parser.add_argument('--gmm_path_08', type=str, required=True)
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
    plant_to_dir = {'axial': 0, 'coronal': 1, 'sagittal': 2}
    trainer_brats19(args, net, current_dir=plant_to_dir[args.plant])

import argparse
import logging
import os
import random
import time
import warnings
import yaml
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from medpy import metric
import torch.backends.cudnn as cudnn

from networks.SDPT import MISSFormer
from datasets.dataset_brats19 import Brats19_dataset

warnings.filterwarnings('ignore')

GREEN = '\033[92m'
END = '\033[0m'

class DynamicGMMSampler:
    def __init__(self, gmm_models_list, device):
        self.device = device
        self.gmm_params = {}
        for l, layer_models in enumerate(gmm_models_list):
            if not layer_models: continue
            self.gmm_params[l] = {}
            for cls_id, moe in layer_models.items():
                w, m, c = [], [], []
                for exp_id, g in moe['experts'].items():
                    w.append(torch.tensor(moe['router'].weights_[exp_id] * g.weights_))
                    m.append(torch.tensor(g.means_))
                    c.append(torch.tensor(g.covariances_))
                cw = torch.cat(w, dim=0)
                cw /= cw.sum()
                self.gmm_params[l][cls_id] = {
                    'weights': cw.to(device),
                    'means': torch.cat(m, dim=0).to(device, dtype=torch.float32),
                    'covs': torch.cat(c, dim=0).to(device, dtype=torch.float32),
                    'pca_comp': torch.tensor(moe['pca'].components_, dtype=torch.float32).to(device),
                    'pca_mean': torch.tensor(moe['pca'].mean_, dtype=torch.float32).to(device),
                    'scaler_scale': torch.tensor(moe['scaler'].scale_, dtype=torch.float32).to(device) if moe[
                        'scaler'] else None,
                    'scaler_mean': torch.tensor(getattr(moe['scaler'], 'mean_', 0), dtype=torch.float32).to(device) if
                    moe['scaler'] else None
                }

    def sample_features(self, l, B, sl, cls_list):
        if l not in self.gmm_params: return None
        valid_cls = [c for c in cls_list if c in self.gmm_params[l]]
        target_c = random.choice(valid_cls) if valid_cls else 0
        p = self.gmm_params[l][target_c]
        idx = torch.multinomial(p['weights'], B * sl, replacement=True)
        z = p['means'][idx] + torch.sqrt(p['covs'][idx] + 1e-6) * torch.randn_like(p['means'][idx])
        z = torch.matmul(z, p['pca_comp']) + p['pca_mean']
        if p['scaler_scale'] is not None: z = z * p['scaler_scale'] + p['scaler_mean']
        return z.view(B, sl, -1)


class RegionDecoupledLossPhase1(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(RegionDecoupledLossPhase1, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_scores = (2. * intersection + smooth) / (union + smooth)


        weighted_dice = (dice_scores[:, 0] + dice_scores[:, 1] + dice_scores[:, 2]) / 3.0
        dice_loss = 1.0 - weighted_dice.mean()
        return self.bce_w * bce_loss + self.dice_w * dice_loss


def train(trainloader, model, decoupled_loss_obj, optimizer, iter_num, current_dir, args, gmm_sampler=None,
          accumulation_steps=4):
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            module.eval()

    loss_sum, gmm_loss_sum = 0.0, 0.0
    run_dice = {"WT": 0.0, "TC": 0.0, "ET": 0.0}

    iterator = tqdm(total=len(trainloader), ncols=160, desc=f"Phase-1 training [direction {current_dir}]")
    optimizer.zero_grad()

    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch[0].squeeze(0).cuda(), sampled_batch[1].squeeze(0).squeeze(1).cuda()
        label_batch[label_batch == 4] = 3

        wt_label = (label_batch > 0).float()
        tc_label = ((label_batch == 1) | (label_batch == 3)).float()
        et_label = (label_batch == 3).float()
        region_targets = torch.stack([wt_label, tc_label, et_label], dim=1)

        B, _, H, W = image_batch.shape
        gmm_l = None
        if args.use_gmm == 1 and gmm_sampler is not None:
            p_cls = []
            if wt_label.sum() > 0: p_cls.append(0)
            if tc_label.sum() > 0: p_cls.append(1)
            if et_label.sum() > 0: p_cls.append(2)
            if len(p_cls) == 0: p_cls = [0]
            seq_l = [(H // (4 * (2 ** l))) * (W // (4 * (2 ** l))) for l in range(4)]
            gmm_l = [gmm_sampler.sample_features(l, B, seq_l[l], p_cls) for l in range(4)]

        model_out = model(image_batch, direction=current_dir, gmm_samples=gmm_l, margin_threshold=args.margin_threshold)

        if isinstance(model_out, tuple):
            outputs = model_out[0]
            total_gmm_loss = model_out[1] if len(model_out) > 1 else None
        else:
            outputs = model_out
            total_gmm_loss = None

        seg_loss = decoupled_loss_obj(outputs, region_targets)

        if args.use_gmm == 1 and total_gmm_loss is not None:
            loss = seg_loss + args.gmm_weight * total_gmm_loss
            gmm_loss_val = total_gmm_loss.item()
        else:
            loss = seg_loss
            gmm_loss_val = 0.0

        (loss / accumulation_steps).backward()

        with torch.no_grad():
            probs = torch.sigmoid(outputs)

            def quick_dice(p, t):
                p = (p > 0.5).float()
                inter = (p * t).sum()
                return (2. * inter + 1e-5) / (p.sum() + t.sum() + 1e-5)

            run_dice["WT"] += quick_dice(probs[:, 0], wt_label).item()
            run_dice["TC"] += quick_dice(probs[:, 1], tc_label).item()
            run_dice["ET"] += quick_dice(probs[:, 2], et_label).item()

        if (i_batch + 1) % accumulation_steps == 0 or (i_batch + 1) == len(trainloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_sum += loss.item()
        gmm_loss_sum += gmm_loss_val

        iterator.update(1)
        iterator.set_postfix({
            'L': f"{loss_sum / (i_batch + 1):.3f}",
            'GMM': f"{gmm_loss_sum / (i_batch + 1):.3f}",
            'WT': f"{run_dice['WT'] / (i_batch + 1):.3f}",
            'TC': f"{run_dice['TC'] / (i_batch + 1):.3f}",
            'ET': f"{run_dice['ET'] / (i_batch + 1):.3f}"
        })

    iterator.close()
    return model, iter_num + 1


def val(valloader, model, decoupled_loss_obj, current_dir):
    model.eval()
    loss_sum = 0.0
    res_dice = {"WT": [], "TC": [], "ET": []}

    iterator = tqdm(total=len(valloader), ncols=150, desc="Validating")
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(valloader):
            image_batch, label_batch = sampled_batch[0].squeeze(0).cuda(), sampled_batch[1].squeeze(0).squeeze(1).cuda()
            label_batch[label_batch == 4] = 3

            model_out = model(image_batch, direction=current_dir)
            outputs = model_out[0] if isinstance(model_out, tuple) else model_out
            probs = torch.sigmoid(outputs)

            wt_t = (label_batch > 0).float()
            tc_t = ((label_batch == 1) | (label_batch == 3)).float()
            et_t = (label_batch == 3).float()
            region_targets = torch.stack([wt_t, tc_t, et_t], dim=1)

            loss_sum += decoupled_loss_obj(outputs, region_targets).item()

            def get_region_dice(p, t):
                p = (p > 0.5).float()
                if t.sum() == 0: return 1.0 if p.sum() == 0 else 0.0
                return metric.binary.dc(p.cpu().numpy(), t.cpu().numpy())

            res_dice["WT"].append(get_region_dice(probs[:, 0], wt_t))
            res_dice["TC"].append(get_region_dice(probs[:, 1], tc_t))
            res_dice["ET"].append(get_region_dice(probs[:, 2], et_t))
            iterator.update(1)

    iterator.close()
    m_wt, m_tc, m_et = np.mean(res_dice["WT"]), np.mean(res_dice["TC"]), np.mean(res_dice["ET"])
    avg_dice = (m_wt + m_tc + m_et) / 3.0

    print(f"\n" + "=" * 50)
    print(f"[Phase-1 validation - direction {current_dir}]")
    print(f"  - WT Dice: {m_wt * 100:.2f}%")
    print(f"  - TC Dice: {m_tc * 100:.2f}%")
    print(f"  - ET Dice: {m_et * 100:.2f}%")
    print(f"  >>> MEAN DICE: {avg_dice * 100:.2f}% <<<")
    print("=" * 50 + "\n")

    return loss_sum / len(valloader), avg_dice, 0.0


def trainer_brats19(args, model, current_dir=0, gmm_sampler=None):
    logging.basicConfig(filename=os.path.join(args.output_dir, "log_phase1.txt"), level=logging.INFO,
                        format='[%(asctime)s] %(message)s')

    db_train = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, root=args.root_path, mode='train',
                               list_file_name=args.train_list)
    db_val = Brats19_dataset(list_dir=args.list_dir, plant=args.plant, root=args.root_path, mode='valid',
                              list_file_name=args.val_list)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    model_module = model.module if hasattr(model, 'module') else model

    if not os.path.isfile(args.resume_ckpt):
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.resume_ckpt}")
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt, map_location='cuda:0', weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        current_model_dict = model_module.state_dict()
        clean_state_dict = {}

        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if k in current_model_dict and v.shape != current_model_dict[k].shape:
                print(
                    f"Skipping a shape-mismatched layer: {k} (checkpoint: {list(v.shape)} -> model: {list(current_model_dict[k].shape)})")
                continue
            clean_state_dict[k] = v

        model_module.load_state_dict(clean_state_dict, strict=False)
        print(f"\nLoaded the shared-backbone checkpoint: {args.resume_ckpt}\n")

    for param in model.parameters():
        param.requires_grad = False

    unfrozen_count = 0
    for name, param in model_module.named_parameters():
        if any(kw in name for kw in [
            f"adapters.{current_dir}.", "last_layer", "aux_head",
            "decoder_0", "decoder_1"
        ]) or (args.use_gmm == 1 and any(kw in name for kw in ["gmm_fusions", "view_translators"])):
            param.requires_grad = True
            unfrozen_count += 1
    print(f"Unfrozen tensors for directional fine-tuning: {unfrozen_count}")


    adapter_params, head_params, decoder_params, gmm_attn_params = [], [], [], []
    for name, param in model_module.named_parameters():
        if not param.requires_grad: continue
        if "last_layer" in name or "aux_head" in name:
            head_params.append(param)
        elif "decoder_0" in name or "decoder_1" in name:
            decoder_params.append(param)
        elif "gmm_fusions" in name or "view_translators" in name:
            gmm_attn_params.append(param)
        else:
            adapter_params.append(param)

    optimizer = optim.AdamW([
        {'params': adapter_params, 'lr': args.adapter_lr},
        {'params': head_params, 'lr': args.head_lr},
        {'params': decoder_params, 'lr': args.decoder_lr},
        {'params': gmm_attn_params, 'lr': args.gmm_lr}
    ], weight_decay=1e-4)

    decoupled_loss_obj = RegionDecoupledLossPhase1(bce_weight=0.5, dice_weight=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    iter_num, best_dice = 0, 0.0

    for epoch in range(args.max_epochs):
        model, iter_num = train(trainloader, model, decoupled_loss_obj, optimizer, iter_num, current_dir, args,
                                gmm_sampler, args.accumulation_steps)
        val_loss, val_dice, val_hd95 = val(valloader, model, decoupled_loss_obj, current_dir)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch} | LR: {current_lr:.6f} | val_dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_path = os.path.join(args.output_dir, f"BEST_DICE_Phase1_DIR{current_dir}.pth")
            torch.save({'model_state_dict': model_module.state_dict(), 'dice': best_dice}, save_path)
            print(f"New best phase-1 mean Dice: {best_dice * 100:.2f}%")
            print(f"Checkpoint saved to: {save_path}")
            print("" * 40 + "\n")

    return "Phase 1 Training Finished!"


def save_args_to_config(args, save_path):
    args_dict = vars(args)
    if not os.path.exists(os.path.dirname(save_path)): os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(args_dict, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: direction-specific adapter training for SDPT")
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root directory containing processed BraTS case folders')
    parser.add_argument('--list_dir', type=str, required=True,
                        help='Directory containing patient-level split files')
    parser.add_argument('--train_list', type=str, required=True,
                        help='Training split filename inside --list_dir')
    parser.add_argument('--val_list', type=str, required=True,
                        help='Validation split filename inside --list_dir')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--plant', type=str, default='sagittal', choices=['axial', 'coronal', 'sagittal'])
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--resume_ckpt', type=str, required=True,
                        help='Shared-backbone checkpoint produced by train_backbone.py')
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--adapter_lr', type=float, default=5e-4)
    parser.add_argument('--head_lr', type=float, default=2e-4)
    parser.add_argument('--decoder_lr', type=float, default=2e-5)
    parser.add_argument('--gmm_lr', type=float, default=1e-4)

    parser.add_argument('--use_gmm', type=int, default=0, choices=[0, 1],
                        help='Enable a single completed-direction GMM prior')
    parser.add_argument('--gmm_path', type=str, default=r'')
    parser.add_argument('--gmm_weight', type=float, default=0.0001)
    parser.add_argument('--margin_threshold', type=float, default=0.65)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed);
    np.random.seed(args.seed);
    torch.manual_seed(args.seed)

    dir_suffix = f"_Phase1_Ultimate_3Channel_GMM{args.use_gmm}_{args.plant}"
    args.output_dir = os.path.join(args.output_dir, time.strftime("%m%d-%H%M") + dir_suffix)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    save_args_to_config(args, os.path.join(args.output_dir, 'train_config.yaml'))

    net = MISSFormer(num_classes=args.num_classes).cuda(0)
    plant_to_dir = {'axial': 0, 'coronal': 1, 'sagittal': 2}

    print("\n" + "=" * 72)
    print("SDPT stage-1 directional adapter training")
    print(f"View: {args.plant}; GMM prior enabled: {bool(args.use_gmm)}")
    print("=" * 72 + "\n")

    gmm_sampler = None
    if args.use_gmm == 1:
        if os.path.exists(args.gmm_path):
            gmm_sampler = DynamicGMMSampler(joblib.load(args.gmm_path), device=torch.device('cuda:0'))
            print(f"{GREEN}Loaded GMM prior bank: {args.gmm_path}{END}")
        else:
            raise FileNotFoundError(f"GMM prior bank not found: {args.gmm_path}")

    trainer_brats19(args, net, current_dir=plant_to_dir[args.plant], gmm_sampler=gmm_sampler)

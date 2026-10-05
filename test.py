
import warnings
import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import yaml
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import SurfaceDiceMetric
from medpy import metric
from datasets.dataset_brats19 import pkload

from networks.SDPT import MISSFormer

warnings.filterwarnings("ignore")
GREEN, RED, YELLOW, CYAN, END = '\033[92m', '\033[91m', '\033[93m', '\033[96m', '\033[0m'


parser = argparse.ArgumentParser(description="Three-view SDPT inference and BraTS evaluation")
parser.add_argument('--root_path', type=str, required=True)
parser.add_argument('--list_dir', type=str, required=True)
parser.add_argument('--test_list', type=str, required=True,
                    help='Evaluation split filename inside --list_dir')
parser.add_argument('--output_dir', type=str, required=True)

parser.add_argument('--num_classes', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--z_spacing', type=float, default=1.0)
parser.add_argument('--margin_threshold', type=float, default=0.65)
parser.add_argument('--fusion_weights', type=str, default='1.2,1.1,1.0',
                    help='Comma-separated axial,coronal,sagittal probability weights')
parser.add_argument('--et_min_size', type=int, default=50)
parser.add_argument('--wt_min_size', type=int, default=100)


parser.add_argument('--axial_checkpoint', type=str, required=True)
parser.add_argument('--coronal_checkpoint', type=str, required=True)
parser.add_argument('--sagittal_checkpoint', type=str, required=True)

parser.add_argument('--gmm_path_ax', type=str, required=True)
parser.add_argument('--gmm_path_cor', type=str, required=True)

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')


class CachedGMMSampler:

    def __init__(self, gmm_models_list, device, pool_size=300000, name="GMM"):
        self.device = device
        self.pool_size = pool_size
        self.feature_pools = {}
        print(f"{CYAN}[{name}] Pre-generating feature pool (size={pool_size})...{END}")

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
                                                                                                0), dtype=torch.float32)
                    features = features * scale + shift
                self.feature_pools[l][cls_id] = features.to(device)
        print(f"{GREEN}[{name}] GMM sampler is ready.{END}")

    def sample_features(self, l, B, sl, cls_list):
        if l not in self.feature_pools: return None
        available_cls = [c for c in cls_list if c in self.feature_pools[l]]
        if not available_cls:
            available_cls = list(self.feature_pools[l].keys())
        target_c = random.choice(available_cls)
        pool = self.feature_pools[l][target_c]

        total_need = B * sl


        if total_need > self.pool_size:
            idx = torch.randint(0, self.pool_size, (total_need,), device=self.device)
            return pool[idx].view(B, sl, -1)
        else:

            start_idx = random.randint(0, self.pool_size - total_need)
            return pool[start_idx: start_idx + total_need].view(B, sl, -1)


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
            current_cls_list = cls_list if l < 2 else ['FG']
            sampler = self.sampler_ax if use_ax else self.sampler_cor
            gmm_l.append(sampler.sample_features(l, B, sls[l], current_cls_list))
        return gmm_l



class BraTSMetricCalculator:
    def __init__(self):
        self.sdcMetric = SurfaceDiceMetric(class_thresholds=[1.0], reduction="mean")

    def get_brats_regions(self, mask):
        et = (mask == 3).astype(np.uint8)
        tc = np.logical_or(mask == 1, mask == 3).astype(np.uint8)
        wt = (mask > 0).astype(np.uint8)
        return [et, tc, wt]

    def calculate_metrics(self, pred, gt, spacing=(1.0, 1.0, 1.0)):
        pred_regions = self.get_brats_regions(pred)
        gt_regions = self.get_brats_regions(gt)
        dsc_list, hd_list, sdc_list = [], [], []
        for p_reg, g_reg in zip(pred_regions, gt_regions):
            if np.sum(g_reg) == 0:
                dsc = 1.0 if np.sum(p_reg) == 0 else 0.0
            else:
                dsc = metric.binary.dc(p_reg, g_reg)
            dsc_list.append(dsc)
            # if np.sum(g_reg) > 0 and np.sum(p_reg) > 0:
            #     try:
            #         hd = metric.binary.hd95(p_reg, g_reg, voxelspacing=spacing)
            #     except:
            #         hd = 373.13
            # else:
            #     hd = 373.13 if (np.sum(p_reg) > 0 or np.sum(g_reg) > 0) else 0.0
            # hd_list.append(hd)
            hd_list.append(0.0)
            if np.sum(g_reg) == 0:
                sdc_val = 1.0 if np.sum(p_reg) == 0 else 0.0
            else:
                p_tensor = torch.from_numpy(p_reg).float().unsqueeze(0).unsqueeze(0)
                g_tensor = torch.from_numpy(g_reg).float().unsqueeze(0).unsqueeze(0)
                sdc_raw = self.sdcMetric(y_pred=p_tensor, y=g_tensor).item()
                sdc_val = 0.0 if np.isnan(sdc_raw) else sdc_raw
            sdc_list.append(sdc_val)
        return dsc_list, sdc_list, hd_list


def load_three_models():
    print(f"\n{GREEN}Loading MISSFormer checkpoints and post-processing settings...{END}")

    def _load(p):
        m = MISSFormer(num_classes=args.num_classes, apply_morph_postprocess=True).to(device)
        ckpt = torch.load(p, map_location=device, weights_only=False)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=False)
        return m.eval()

    return _load(args.axial_checkpoint), _load(args.coronal_checkpoint), _load(args.sagittal_checkpoint)



def predict_volume_single(model, image_3d, plant, adapter_dir, sw_batch_size, gmm_sampler=None):
    model.eval()
    model.apply_morph_postprocess = False

    if plant == 'axial':
        test_input = torch.from_numpy(image_3d).float().permute(2, 3, 0, 1)
    elif plant == 'coronal':
        test_input = torch.from_numpy(image_3d).float().permute(1, 3, 0, 2)
    elif plant == 'sagittal':
        test_input = torch.from_numpy(image_3d).float().permute(0, 3, 1, 2)

    test_input = test_input.to(device).contiguous()


    def _predictor(x):
        B_chunk = x.shape[0]
        gmm_l = None

        if gmm_sampler is not None:

            sls = [2304, 576, 144, 36]

            if isinstance(gmm_sampler, DualGMMSampler):
                gmm_l = gmm_sampler.sample_features_all_layers(B_chunk, sls, [0, 1, 2])
            else:

                gmm_l = []
                for l in range(4):
                    cls_pool = [0, 1, 2] if l < 2 else ['FG']
                    gmm_l.append(gmm_sampler.sample_features(l, B_chunk, sls[l], cls_pool))


        outputs = model(x, direction=adapter_dir, gmm_samples=gmm_l, margin_threshold=args.margin_threshold)

        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        return torch.sigmoid(logits)

    with torch.no_grad():
        prob_stack = sliding_window_inference(
            inputs=test_input, roi_size=(192, 192), sw_batch_size=sw_batch_size,
            predictor=_predictor, overlap=0.3, mode="gaussian"
        ).cpu().numpy()

    if plant == 'axial':
        res_prob = np.transpose(prob_stack, (1, 2, 3, 0))
    elif plant == 'coronal':
        res_prob = np.transpose(prob_stack, (1, 2, 0, 3))
    elif plant == 'sagittal':
        res_prob = np.transpose(prob_stack, (1, 0, 2, 3))
    return res_prob


def inference_ensemble_ultimate(model_axial, model_coronal, model_sagittal, ax_sampler, dual_sampler):
    metric_calc = BraTSMetricCalculator()
    stats = {k: [] for k in ['et_dice', 'tc_dice', 'wt_dice', 'et_sdc', 'tc_sdc', 'wt_sdc']}

    eval_file_path = os.path.join(args.list_dir, args.test_list)
    with open(eval_file_path, 'r') as f:
        case_names = [l.strip() for l in f if l.strip()]


    os.makedirs(args.output_dir, exist_ok=True)
    network_name = "Ours_Ensemble"
    eval_filename_base = os.path.splitext(os.path.basename(eval_file_path))[0]
    out_txt_path = os.path.join(args.output_dir, f"{network_name}_{eval_filename_base}_eval_results.txt")
    print(f"{YELLOW}Writing case-wise metrics to: {os.path.abspath(out_txt_path)}{END}\n")

    w = np.asarray([float(value) for value in args.fusion_weights.split(',')], dtype=np.float32)
    if w.shape != (3,) or np.any(w < 0) or float(w.sum()) <= 0:
        raise ValueError('--fusion_weights must contain three non-negative values with a positive sum')


    with open(out_txt_path, 'w') as f_out:
        header = f"{'Case_Name':<35} | {'Avg_DSC':<8} | {'Avg_SDC':<8} | {'DSC_ET':<8} | {'SDC_ET':<8} | {'DSC_TC':<8} | {'SDC_TC':<8} | {'DSC_WT':<8} | {'SDC_WT':<8}\n"
        f_out.write(header)
        f_out.write("-" * len(header) + "\n")

        for case_idx, case_name in enumerate(tqdm(case_names)):
            actual_name = case_name.replace('\\', '/').split('/')[-1]
            pkl_path = os.path.join(args.root_path, case_name, f"{actual_name}_pkl_ui8f32b0.pkl")
            if not os.path.exists(pkl_path): continue
            image_3d_raw, label_3d_raw = pkload(pkl_path)

            image_3d_pad = np.pad(image_3d_raw, ((0, 0), (0, 0), (20, 20), (0, 0)), mode='constant')
            label_3d = np.pad(label_3d_raw, ((0, 0), (0, 0), (20, 20)), mode='constant')
            label_3d[label_3d == 4] = 3

            p_ax = predict_volume_single(model_axial, image_3d_pad, 'axial', 0, args.batch_size, gmm_sampler=None)

            p_co = predict_volume_single(model_coronal, image_3d_pad, 'coronal', 1, args.batch_size, gmm_sampler=ax_sampler)
  
            p_sa = predict_volume_single(model_sagittal, image_3d_pad, 'sagittal', 2, args.batch_size,
                                         gmm_sampler=dual_sampler)

            f_prob_final = (p_ax * w[0] + p_co * w[1] + p_sa * w[2]) / np.sum(w)

            prob_wt, prob_tc, prob_et = f_prob_final[0], f_prob_final[1], f_prob_final[2]
            pred_3d = np.zeros_like(prob_wt, dtype=np.uint8)
            pred_3d[prob_wt > 0.50] = 2
            pred_3d[prob_tc > 0.50] = 1
            pred_3d[prob_et > 0.50] = 3

            import scipy.ndimage as ndimage
            from skimage import morphology
            tc_mask = (pred_3d == 1) | (pred_3d == 3)
            pred_3d[ndimage.binary_fill_holes(tc_mask) & (~tc_mask)] = 1
            et_mask = (pred_3d == 3)
            et_cleaned = morphology.remove_small_objects(et_mask, min_size=args.et_min_size)
            pred_3d[et_mask & (~et_cleaned)] = 1
            wt_mask = (pred_3d > 0)
            wt_cleaned = morphology.remove_small_objects(wt_mask, min_size=args.wt_min_size)
            pred_3d[wt_mask & (~wt_cleaned)] = 0

            dscs, sdcs, _ = metric_calc.calculate_metrics(pred_3d, label_3d, spacing=(1.0, 1.0, args.z_spacing))
            stats['et_dice'].append(dscs[0])
            stats['tc_dice'].append(dscs[1])
            stats['wt_dice'].append(dscs[2])
            stats['et_sdc'].append(sdcs[0])
            stats['tc_sdc'].append(sdcs[1])
            stats['wt_sdc'].append(sdcs[2])

            curr_avg_dsc = np.mean(dscs)
            curr_avg_sdc = np.mean(sdcs)


            f_out.write(f"{actual_name:<35} | {curr_avg_dsc:.4f}   | {curr_avg_sdc:.4f}   | {dscs[0]:.4f}   | {sdcs[0]:.4f}   | {dscs[1]:.4f}   | {sdcs[1]:.4f}   | {dscs[2]:.4f}   | {sdcs[2]:.4f}\n")
            f_out.flush()

            print(f"{YELLOW}Case {case_idx} ({actual_name}){END} -> "
                  f"ET: {dscs[0] * 100:.2f}% | TC: {dscs[1] * 100:.2f}% | WT: {dscs[2] * 100:.2f}%")


        f_out.write("-" * len(header) + "\n")
        e_m_d, t_m_d, w_m_d = np.mean(stats['et_dice']), np.mean(stats['tc_dice']), np.mean(stats['wt_dice'])
        e_m_s, t_m_s, w_m_s = np.mean(stats['et_sdc']), np.mean(stats['tc_sdc']), np.mean(stats['wt_sdc'])
        m_m_d = (e_m_d + t_m_d + w_m_d) / 3.0
        m_m_s = (e_m_s + t_m_s + w_m_s) / 3.0
        f_out.write(f"{'OVERALL_MEAN':<35} | {m_m_d:.4f}   | {m_m_s:.4f}   | {e_m_d:.4f}   | {e_m_s:.4f}   | {t_m_d:.4f}   | {t_m_s:.4f}   | {w_m_d:.4f}   | {w_m_s:.4f}\n")

    print("\n" + "=" * 95)
    print(f"{RED}=== GMM-guided three-view inference summary ==={END}")
    print(f"{'Metric':<10} | {'ET (Enhancing)':^25} | {'TC (Core)':^25} | {'WT (Whole)':^25}")
    print("-" * 95)
    for label, m_key, factor in [('Dice (%)', 'dice', 100.0), ('SDC (%)', 'sdc', 100.0)]:
        e_data = np.array(stats[f'et_{m_key}']) * factor
        t_data = np.array(stats[f'tc_{m_key}']) * factor
        w_data = np.array(stats[f'wt_{m_key}']) * factor
        print(f"{GREEN}{label:<10}{END} | "
              f"{np.mean(e_data):6.2f} ± {np.std(e_data):5.2f} | "
              f"{np.mean(t_data):6.2f} ± {np.std(t_data):5.2f} | "
              f"{np.mean(w_data):6.2f} ± {np.std(w_data):5.2f}")
    total_dice = (np.mean(stats['et_dice']) + np.mean(stats['tc_dice']) + np.mean(stats['wt_dice'])) / 3.0 * 100
    total_sdc = (np.mean(stats['et_sdc']) + np.mean(stats['tc_sdc']) + np.mean(stats['wt_sdc'])) / 3.0 * 100

    print("-" * 95)
    print(f"{YELLOW}Global mean Dice: {total_dice:.2f}%{END}")
    print(f"{YELLOW}Global mean SDC: {total_sdc:.2f}%{END}")
    print(f"{GREEN}Detailed metrics saved to: {os.path.abspath(out_txt_path)}{END}\n")


if __name__ == "__main__":
    required_paths = [
        args.axial_checkpoint, args.coronal_checkpoint, args.sagittal_checkpoint,
        args.gmm_path_ax, args.gmm_path_cor,
        os.path.join(args.list_dir, args.test_list),
    ]
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required input not found: {path}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'evaluation_config.yaml'), 'w', encoding='utf-8') as stream:
        yaml.safe_dump(vars(args), stream, sort_keys=False, allow_unicode=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


    m_ax, m_co, m_sa = load_three_models()
    total_params = sum(p.numel() for p in m_ax.parameters())
    print(f"{CYAN}Model parameters: {total_params / 1e6:.2f} M{END}\n")

    ax_sampler = CachedGMMSampler(joblib.load(args.gmm_path_ax), device=device, name="Axial GMM")
    cor_sampler = CachedGMMSampler(joblib.load(args.gmm_path_cor), device=device, name="Coronal GMM")
    dual_sampler = DualGMMSampler(ax_sampler, cor_sampler, mode='random')

    inference_ensemble_ultimate(m_ax, m_co, m_sa, ax_sampler, dual_sampler)

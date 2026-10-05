# Sequential Directional Prior Transfer (SDPT)

Code and model assets for **Sequential Directional Prior Transfer for Multi-View 2.5D Brain Tumor Segmentation in Multimodal MRI**.

SDPT transfers information sequentially from axial to coronal to sagittal views using direction-specific adapters, reconstructed Gaussian-mixture priors, view alignment, and uncertainty-aware fusion.

Repository: <https://github.com/NuistSMS/SDPT>

## Package contents

This README describes the files in `SDPT_PAA_Reproducibility_Package`. Run the commands from that directory, or from the repository root after uploading these contents.

```text
SDPT_PAA_Reproducibility_Package/
├── README.md
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── THIRD_PARTY_NOTICE.md
├── MANIFEST.sha256
├── train_backbone.py                 # shared backbone initialization
├── trainer_backbone.py              # backbone training and validation
├── train-first.py                   # first directional stage
├── train-secoce.py                  # second stage with one source prior bank
├── trian-third.py                   # sagittal stage with axial/coronal priors
├── test.py                          # three-view volumetric evaluation
├── GMM.py                           # fit source-direction GMM banks
├── utils.py
├── networks/
│   ├── MISSFOREMR.py                # backbone model
│   ├── SDPT.py                      # directional SDPT model
│   ├── segformer.py
│   └── __init__.py
├── datasets/
│   ├── dataset_brats19.py           # serialized-case loader for both datasets
│   └── __init__.py
├── configs/
│   ├── brats2019_reference.yaml
│   └── brats2021_reference.yaml
├── lists/
│   ├── BraTS2019/
│   │   ├── t4.txt
│   │   └── v4.txt
│   └── BraTS2021/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       ├── p21.txt
│       └── ptest.txt
├── checkpoints/
│   ├── model_out_aixl_19/axial.pth
│   ├── model_out_cor19/cor.pth
│   └── model_out_sagittal19/sag.pth
└── gmm_banks/
    ├── BraTS2019/
    │   ├── LKA_GMM_Final_axial.pkl
    │   ├── LKA_GMM_Final_0.8axial.pkl
    │   ├── LKA_GMM_Final_0.5coronal.pkl
    │   └── LKA_GMM_Final_0.8coronal.pkl
    └── BraTS2021/
        ├── axial_ratio0.5.pkl
        ├── axial_ratio0.8.pkl
        ├── coronal_ratio0.5.pkl
        └── coronal_ratio0.8.pkl
```

The spellings `train-secoce.py`, `trian-third.py`, and `model_out_aixl_19` are the actual filenames. Use them exactly as shown.

Three BraTS2019 directional checkpoint files, eight fitted GMM banks, and the GMM-fitting entry point are included. Raw MRI data, BraTS2021 checkpoints, a standalone backbone checkpoint, and archived evaluation results are not included in this folder. There is no `scripts/` or `results/` directory in this release.

The backbone is imported from `networks.MISSFOREMR`, and all directional training, evaluation, and GMM fitting import from `networks.SDPT`. Both modules retain the class name `MISSFormer`; the filename alignment does not change network parameters or checkpoint keys. `MISSFOREMR.py` is the exact filename, including its spelling.

## Environment

The provided environment specification uses Python 3.12.7 and the pinned dependencies in `requirements.txt`, including PyTorch 2.9.1, TorchVision 0.24.1, MONAI 1.5.1, and scikit-learn 1.5.1.

```bash
conda env create -f environment.yml
conda activate sdpt
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

An existing compatible Python environment can instead install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The CUDA 13.0 note in `environment.yml` records the captured development environment; it does not install a GPU driver or guarantee the CUDA variant of the installed PyTorch wheel. Training scripts use CUDA explicitly and require a working CUDA-enabled PyTorch environment. The evaluation script has a CPU fallback, but full-volume evaluation and GMM feature pools can be memory-intensive.

Command examples below use Bash line continuations (`\`). In Windows PowerShell, enter each command on one line or replace the continuations with backticks. Replace `/data/...` and `/path/to/...` with your local paths.

## Data preparation and splits

Obtain the MRI data and labels from the official BraTS providers under their applicable terms. This package reads preprocessed pickle files, not NIfTI files directly. Each file must contain `(image, label)`, where `image` is a floating-point NumPy array with shape `(H, W, D, 4)` and `label` is an integer array with shape `(H, W, D)`.

Data preprocessing follows SuperLightNet. For training, labels are background `0`, necrotic/non-enhancing core `1`, edema `2`, and enhancing tumor `3`. The evaluation script also remaps BraTS label `4` to `3`.

The required path for each split entry is:

```text
<root_path>/<split_entry>/<case_name>_pkl_ui8f32b0.pkl
```

For example:

```text
/data/BraTS2019/processed/HGG/BraTS19_CBICA_ALN_1/BraTS19_CBICA_ALN_1_pkl_ui8f32b0.pkl
/data/BraTS2021/processed/BraTS2021_01639/BraTS2021_01639_pkl_ui8f32b0.pkl
```

Keep the `HGG/` and `LGG/` prefixes in BraTS2019 list entries. The loader pads the third spatial dimension by 20 voxels on each side and center-crops to `192 × 192 × 192` for training and validation before sampling directional slices. Evaluation uses sliding windows of `192 × 192` on the padded volume.

| Dataset | Files present | Number of entries |
|---|---|---|
| BraTS2019 | `t4.txt` / `v4.txt` | 268 / 67 |
| BraTS2021 | `train.txt` / `val.txt` / `test.txt` | 875 / 125 / 251 |
| BraTS2021 small subsets | `p21.txt` / `ptest.txt` | 6 / 6 |

Only the fourth BraTS2019 partition is packaged. The reference YAML describes a five-fold protocol, but the other four partitions are absent. The six-case lists are not substitutes for the quantitative evaluation split.

Training scripts select checkpoints using `--val_list`. If `v4.txt` is used both for checkpoint selection and final reporting, those results must not be described as evaluation on a separate untouched test set. Confirm that the checkpoint and prior-bank training cases match the intended partition before interpreting evaluation scores.

## Evaluate the included BraTS2019 checkpoints

After preparing the processed cases listed in `v4.txt`, the following command uses paths that exist in this package:

```bash
python test.py \
  --root_path /data/BraTS2019/processed \
  --list_dir lists/BraTS2019 --test_list v4.txt \
  --output_dir outputs/brats2019/evaluation \
  --axial_checkpoint checkpoints/model_out_aixl_19/axial.pth \
  --coronal_checkpoint checkpoints/model_out_cor19/cor.pth \
  --sagittal_checkpoint checkpoints/model_out_sagittal19/sag.pth \
  --gmm_path_ax gmm_banks/BraTS2019/LKA_GMM_Final_0.8axial.pkl \
  --gmm_path_cor gmm_banks/BraTS2019/LKA_GMM_Final_0.8coronal.pkl \
  --fusion_weights 1.2,1.1,1.0 --margin_threshold 0.65 \
  --et_min_size 50 --wt_min_size 100 \
  --batch_size 48 --seed 1234 --gpu_id 0
```

This is an invocation for the available files, not a verified checkpoint-to-fold or checkpoint-to-paper-table mapping. A checkpoint's filename alone does not establish its training split.

The script writes `evaluation_config.yaml` and `Ours_Ensemble_v4_eval_results.txt` to the output directory. It reports case-wise and overall DSC and SDC for ET, TC, and WT. The text file uses fractions; the terminal summary displays percentages. HD95 computation is disabled in the current evaluator, so it does not produce meaningful HD95 results.

The executable evaluation settings are:

- Weighted axial/coronal/sagittal probability fusion at `1.2/1.1/1.0`, followed by a `0.5` region threshold.
- Tumor-core hole filling and removal of enhancing-tumor components below 50 voxels and whole-tumor components below 100 voxels.
- Surface Dice tolerance of one voxel. `--z_spacing` does not alter the current SDC calculation.
- Random sampling of reconstructed GMM features, with the supplied seed applied before model loading and inference.

Before interpreting a run, verify that all listed case files exist: `test.py` currently skips missing case files. For `v4.txt`, the result file should contain 67 case rows. Reducing `--batch_size` may help with GPU memory use, but GMM feature pools also consume memory, and changing batching can change the random sampling sequence.

For BraTS2021, use `lists/BraTS2021/test.txt` and its two ratio-0.8 banks, together with the corresponding three BraTS2021 checkpoints. Those checkpoints are not present in this folder.

## Training entry points

Training follows this order:

```text
Shared backbone → axial adaptation → fit axial banks
                → coronal adaptation → fit coronal banks
                → sagittal adaptation → three-view evaluation
```

The YAML files under `configs/` document reference settings; the training programs do not load them automatically. Supply the relevant command-line arguments explicitly. The examples below use the complete BraTS2021 train/validation lists. Paths under `/path/to/` denote checkpoints that must be supplied or selected from a preceding training run.

### 1. Shared backbone initialization

```bash
python train_backbone.py \
  --root_path /data/BraTS2021/processed \
  --list_dir lists/BraTS2021 --train_list train.txt --val_list val.txt \
  --output_dir outputs/brats2021/backbone \
  --max_epochs 120 --batch_size 36 --base_lr 0.0003 \
  --plant all --seed 1234 --gpu_id 0
```

The legacy dataset identifier defaults to `Brats19` even when supplied BraTS2021 paths; it is the supported identifier in this entry point. The script appends a timestamp to the output directory and writes selected checkpoints under `best_models/`. Choose a checkpoint using the designated validation partition.

### 2. Axial adaptation

```bash
python train-first.py \
  --root_path /data/BraTS2021/processed \
  --list_dir lists/BraTS2021 --train_list train.txt --val_list val.txt \
  --output_dir outputs/brats2021/axial \
  --resume_ckpt /path/to/selected_backbone.pth \
  --plant axial --max_epochs 150 --use_gmm 0 \
  --seed 1234 --gpu_id 0
```

Set `--plant axial` explicitly: the first-stage script defaults to `sagittal`. The axial checkpoint is saved as `BEST_DICE_Phase1_DIR0.pth` in the specified output directory.

### 3. Coronal adaptation using axial priors

```bash
python train-secoce.py \
  --root_path /data/BraTS2021/processed \
  --list_dir lists/BraTS2021 --train_list train.txt --val_list val.txt \
  --output_dir outputs/brats2021/coronal \
  --resume_ckpt outputs/brats2021/axial/BEST_DICE_Phase1_DIR0.pth \
  --plant coronal --max_epochs 100 --switch_epoch 60 \
  --gmm_path_05 gmm_banks/BraTS2021/axial_ratio0.5.pkl \
  --gmm_path_08 gmm_banks/BraTS2021/axial_ratio0.8.pkl \
  --gmm_weight 0.05 --margin_threshold 0.65 --seed 1234 --gpu_id 0
```

### 4. Sagittal adaptation using both source directions

```bash
python trian-third.py \
  --root_path /data/BraTS2021/processed \
  --list_dir lists/BraTS2021 --train_list train.txt --val_list val.txt \
  --output_dir outputs/brats2021/sagittal \
  --resume_ckpt /path/to/selected_coronal.pth \
  --plant sagittal --max_epochs 10 --switch_epoch 6 \
  --gmm_path_ax_05 gmm_banks/BraTS2021/axial_ratio0.5.pkl \
  --gmm_path_ax_08 gmm_banks/BraTS2021/axial_ratio0.8.pkl \
  --gmm_path_cor_05 gmm_banks/BraTS2021/coronal_ratio0.5.pkl \
  --gmm_path_cor_08 gmm_banks/BraTS2021/coronal_ratio0.8.pkl \
  --gmm_weight 0.05 --margin_threshold 0.65 --seed 1234 --gpu_id 0
```

These commands demonstrate the available training interfaces with the supplied banks. A new end-to-end training run needs banks fitted from its own selected source-direction checkpoints and training cases, using `GMM.py` as shown below. Using existing banks with newly trained checkpoints does not establish equivalence to the original experiment.

### Fitting GMM banks between directional stages

After axial adaptation, fit an axial bank using only the training split:

```bash
python GMM.py \
  --model_path outputs/brats2021/axial/BEST_DICE_Phase1_DIR0.pth \
  --root_path /data/BraTS2021/processed \
  --list_dir lists/BraTS2021 --train_list train.txt \
  --output_gmm_dir outputs/brats2021/gmm \
  --plant axial --adapter_dir 0 --boundary_ratio 0.5 --gpu_id 0
```

Repeat with `--boundary_ratio 0.8`. The outputs are `LKA_GMM_Final_0.5_axial.pkl` and `LKA_GMM_Final_0.8_axial.pkl` in the selected output directory. Pass these new files as `--gmm_path_05` and `--gmm_path_08` when training the coronal stage.

After coronal adaptation, fit the coronal bank:

```bash
python GMM.py \
  --model_path /path/to/selected_coronal.pth \
  --root_path /data/BraTS2021/processed \
  --list_dir lists/BraTS2021 --train_list train.txt \
  --output_gmm_dir outputs/brats2021/gmm \
  --plant coronal --adapter_dir 1 --boundary_ratio 0.5 --gpu_id 0
```

Repeat with `--boundary_ratio 0.8`. This produces `LKA_GMM_Final_0.5_coronal.pkl` and `LKA_GMM_Final_0.8_coronal.pkl`. For sagittal training, replace the four supplied-bank paths in the example with the corresponding newly fitted axial and coronal files. For evaluation, use the newly fitted ratio-0.8 files.

For BraTS2019 fitting, use its processed root, `--list_dir lists/BraTS2019`, and `--train_list t4.txt`. Do not use validation or test lists for fitting. Direction indices are axial `0`, coronal `1`, and sagittal `2`. The fitting entry point samples features from up to 100 batches of eight cases; its sampling is stochastic, so refitting need not produce byte-identical banks. Existing packaged banks retain their original filenames.

For BraTS2019, use `lists/BraTS2019`, `t4.txt`, `v4.txt`, and the corresponding dataset root. The reference configuration specifies 300 backbone epochs and 150 sagittal epochs. It does not specify a BraTS2019 sagittal bank-switch epoch. The file `LKA_GMM_Final_axial.pkl` has no ratio in its name; confirm its provenance before treating it as a ratio-0.5 bank.

Directional fine-tuning freezes parameters first, then enables the current adapter, segmentation heads, and `decoder_0`/`decoder_1`; the GMM stages additionally enable fusion modules and view translators. Thus the executable schedule updates more than the adapters alone.

## Releasing files on GitHub

GMM banks are explicitly allowed by `.gitignore`. Preserve their directory names and check file-size restrictions when uploading. Regenerate `MANIFEST.sha256` whenever the release contents change. Load pickle, joblib, and PyTorch checkpoint files only from trusted sources.

## Citation and third-party code

Citation metadata are in `CITATION.cff`. Cite the associated paper once a published bibliographic record is available.

The network implementation builds on MISSFormer. See `THIRD_PARTY_NOTICE.md` for attribution and redistribution terms. BraTS images and labels are not included.

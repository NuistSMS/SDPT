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
│   ├── MISSFOREMR.py         # backbone model
│   ├── SDPT.py                 # directional SDPT model
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

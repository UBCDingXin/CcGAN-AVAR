# [Imbalance-Robust and Sampling-Efficient Continuous Conditional GANs via Adaptive Vicinity and Auxiliary Regularization](https://arxiv.org/abs/2508.01725)

--------------------------------------------------------
If you use this code, please cite
```text
@article{ding2025imbalance,
  title={Imbalance-Robust and Sampling-Efficient Continuous Conditional GANs via Adaptive Vicinity and Auxiliary Regularization},
  author={Ding, Xin and Chen, Yun and Wang, Yongwei and Zhang, Kao and Zhang, Sen and Cao, Peibei and Wang, Xiangxue},
  journal={arXiv preprint arXiv:2508.01725},
  year={2025}
}
```

--------------------------------------------------------
## To do list:  
- [x] Support training CcGAN and CcGAN-AVAR on multiple datasets with a unified framework. 
- [x] Support DCGAN, SNGAN, SAGAN, BigGAN and BigGAN-deep architectures.
- [x] Support three types of label embeeding: CcGAN's ILI, Sinusoidal, and Gaussian Fourier.
- [x] Support mixed precision training based on Accelerate. 
- [x] Support Exponential Moving Average (EMA). Not compatible with self-attention in SAGAN and BigGAN!
- [x] Support Muilti-GPU training (refer to https://github.com/NUIST-GenAI-Lab/CcGAN-AVAR_DDP).

--------------------------------------------------------
## Software Requirements
| Item | Version |
|---|---|
| OS | Ubuntu 22.04 |
| CUDA  | 12.8 |
| MATLAB | R2021 |
| Python | 3.12.7 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| h5py | 3.11.0 |
| matplotlib | 3.9.2 |
| Pillow | 10.4.0 |
| torch | 2.7.0 |
| torchvision | 0.22.0 |
| accelearate | 1.6.0 |


All 256x256 experiments need to be conducted with torch>=2.8.


--------------------------------------------------------
## Datasets
#### RC-49 (64x64)
[RC-49_64x64_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstI0OuDMqpEZA80tRQ?e=fJJbWw) <br />
[RC-49_64x64_BaiduYun_link](https://pan.baidu.com/s/1Odd02zraZI0XuqIj5UyOAw?pwd=bzjf) <br />

#### RC-49-I (64x64)
[RC-49-I_64x64_OneDrive_link](https://1drv.ms/u/c/907562db44a4f6b8/EbJrU1Vc_p9BjgSOeKS8QUgBOZLbGTBsnShRGLXlRC516g?e=scNBPW) <br />
[RC-49-I_64x64_BaiduYun_link](https://pan.baidu.com/s/1DgVy_AdQgFVVRbmTleggrQ?pwd=qfud) <br />

### The preprocessed UTKFace Dataset (h5 file)
#### UTKFace (64x64)
[UTKFace_64x64_Onedrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstIzurW-LCFpGz5D7Q?e=X23ybx) <br />
[UTKFace_64x64_BaiduYun_link](https://pan.baidu.com/s/1fYjxmD3tJG6QKw5jjXxqIg?pwd=ocmi) <br />
#### UTKFace (128x128)
[UTKFace_128x128_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstJGpTgNYrHE8DgDzA?e=d7AeZq) <br />
[UTKFace_128x128_BaiduYun_link](https://pan.baidu.com/s/17Br49DYS4lcRFzktfSCOyA?pwd=iary) <br />
#### UTKFace (192x192)
[UTKFace_192x192_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstY8hLN3lWEyX0lNLA?e=BcjUQh) <br />
[UTKFace_192x192_BaiduYun_link](https://pan.baidu.com/s/1KaT_k21GTdLqqJxUi24f-Q?pwd=4yf1) <br />
#### UTKFace (256x256)
[UTKFace_256x256_OneDrive_link](https://1drv.ms/u/c/907562db44a4f6b8/EaWxQlfC3nVFlxnLDPRIjLkB5i9t6UYXHG40E0Ms2u0ZvQ?e=xL6MuJ) <br />
[UTKFace_256x256_BaiduYun_link](https://pan.baidu.com/s/1uX-_kafmGVc-1Ox_HNfAxg?pwd=wuvm) <br />

### The Steering Angle dataset (h5 file)
#### Steering Angle (64x64)
[SteeringAngle_64x64_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstIyDTDpGA0CNiONkA?e=Ui5kUK) <br />
[SteeringAngle_64x64_BaiduYun_link](https://pan.baidu.com/s/1ekpMJLC0mE08zVJp5GpFHQ?pwd=xucg) <br />
#### Steering Angle (128x128)
[SteeringAngle_128x128_OneDrive_link](https://1drv.ms/u/s!Arj2pETbYnWQstJ0j7rXhDtm6y4IcA?e=bLQh2e) <br />
[SteeringAngle_128x128_BaiduYun_link](https://pan.baidu.com/s/1JVBccsr5vgsIdzC-uskx-A?pwd=4z5n) <br />
#### Steering Angle (256x256)
[SteeringAngle_256x256_OneDrive_link](https://1drv.ms/u/c/907562db44a4f6b8/Ed29A3YeV4NMjo-4qSFS8G0BlQpMUB4D0V_xNin8KpQIVQ?e=Ijztt6) <br />
[SteeringAngle_256x256_BaiduYun_link](https://pan.baidu.com/s/1bSQO7c47F0fIlEhmQ95poA?pwd=mkxz) <br />

--------------------------------------------------------
## Preparation (Required!)
Download the evaluation checkpoints (zip file) from [OneDrive](https://1drv.ms/u/c/907562db44a4f6b8/EZQMkKev3alAh2gsqWx01zABDdJCLVKWTal-vjc_uwk2vA?e=Bbnu65) or [BaiduYun](https://pan.baidu.com/s/1wbN5_0CZTe1Ko3KwTWiwIg?pwd=mptb), then extract the contents to `./CcGAN-AVAR/evaluation/eval_ckpts`.

--------------------------------------------------------
## Training

### (1) Auxiliary regression model training
Before training CcGAN-AVAR, first train the auxiliary ResNet18 regression model by executing the `.sh` scripts in `./config/aux_reg`. Ensure the root path and data path are correctly configured.

### (2) CcGAN-AVAR training
We provide the `.sh` file for training CcGAN-AVAR-S or CcGAN-AVAR-H on each dataset in `./config`. Ensure the root path and data path are correctly configured.

--------------------------------------------------------
## Sampling and Evaluation

<!------------------------------------>
### (1) SFID, Diversity, and Label Score
After the training, the sampling usually automatically starts. Ensure that the `--do_eval` flag is enabled. 

<!------------------------------------>
### (2) NIQE
To enable NIQE calculation, set both `--dump_fake_for_niqe` and `--niqe_dump_path` to output generated images to your specified directory. Implementation details are available at: https://github.com/UBCDingXin/CCDM

--------------------------------------------------------
## Acknowledge
- https://github.com/UBCDingXin/improved_CcGAN
- https://github.com/UBCDingXin/Dual-NDA
- https://github.com/UBCDingXin/CCDM
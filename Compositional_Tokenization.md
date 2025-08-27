# Compositional Motion Tokenization (VQ-VAE Training)

This document provides detailed information about the Compositional Motion Tokenization stage of the Language of Motion (LoM) framework, including training procedures, metrics, and results.

## Overview

Compositional Motion Tokenization is the first stage of the LoM training pipeline. It involves training separate VQ-VAE models for different body regions to learn discrete motion representations. This stage decomposes the human body into four main regions:

- **Face**: Facial expressions and yaw movements
- **Hands**: Hand gestures and finger movements  
- **Upper Body**: Torso, arms, and shoulder movements
- **Lower Body**: Legs, feet, and hip movements

## Training Commands

Face Region:
```bash
python -m train --cfg configs/config_mixed_stage1_vq_face_256_512_ds4_wo_mesh_lr1e-4.yaml --nodebug
```

Upper Body Region:
```bash
python -m train --cfg configs/config_mixed_stage1_vq_upper_256_512_ds4_wo_mesh_lr1e-4.yaml --nodebug
```

Lower Body Region:
```bash
python -m train --cfg configs/config_mixed_stage1_vq_lower_128_512_ds4_wo_mesh_lr1e-4.yaml --nodebug
```

Hand Region:
```bash
python -m train --cfg configs/config_mixed_stage1_vq_hand_256_512_ds4_wo_mesh_lr1e-4.yaml --nodebug
```


## Evaluation (Reconstruction Metrics)


Run the following command can get the Reconstruction Metrics(MPJPE,MPVPE, PAMPJPE etc).:
```bash
python -m test --cfg configs/config_mixed_stage1_vq_compositional.yaml
```
> NOTE: Update the following fields in `config_mixed_stage1_vq_compositional.yaml`:
> - `CHECKPOINTS_FACE`
> - `CHECKPOINTS_HAND`
> - `CHECKPOINTS_UPPER`
> - `CHECKPOINTS_LOWER`
> - `code_num`
> - `codebook_size`
> 
> Replace them with your own checkpoints.  
> We also provide pretrained checkpoints on Hugging Face.
>
> All checkpoints reported here were trained on AMASS and BEAT2 datasets to ensure stronger performance.


Here we report the following metrics to comunity for further research, all the checkpoints were trained on AMASS and BEAT2 dataset to try better performance:

### Extra quantitative evaluation on BEAT2 test sets (all speakers)

| Upper Size + Dim | Lower Size + Dim | DS | MPJPE | PA-MPJPE | MPVPE | ACCEL |
|------------------|------------------|----|-------|----------|-------|-------|
|    512 + 256     |    512 + 128     | 4  | 53.97 |  33.54   | 63.49 |  1.24 |
|    512 + 256     |    512 + 256     | 4  | 53.65 |  33.85   | 62.47 |  1.42 |



| Face Size + Dim | DS |  LVE   |   FFD  | MPVPE-FLAME |
|-----------------|----|--------|--------|-------------|
|    512 + 256    | 4  |2.25e-06|8.94e-08|   0.156     |


> NOTE: We fixed the head rotation, so the number is pretty small than usual. If you want to use this metrics in your paper, please be careful on this part!!

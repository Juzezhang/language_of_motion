# The Language of Motion

[![arXiv](https://img.shields.io/badge/arXiv-2412.10523-b31b1b.svg)](https://arxiv.org/pdf/2412.10523)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://languageofmotion.github.io/)

This repository contains the official implementation of "The Language of Motion: Unifying Verbal and Non-verbal Language of 3D Human Motion".

## 🔍 Overview

Language of Motion (LoM) is a framework that models human motion generation as a sequence modeling problem using language models. It decomposes the human body into separate regions (face, hands, upper, and lower body) to effectively capture and generate natural human movements from various modalities such as text and audio.

![Teaser](./assets/teaser.png)

## ✅ TODO List

- [x] Initial code release
- [x] Inference code for text-to-motion
- [x] Inference code for co-speech gesture generation
- [x] Tokenizer training code
- [x] AMASS and LibriSpeech preprocessing code
- [x] Evaluation Benchmark results
- [ ] Text-to-motion Result on rotation format
- [x] Language model training code


## 🛠️ Environment Setup
<details>

We use Conda for environment management. Follow these steps to set up the development environment:

```bash
# Create and activate the conda environment
conda create --name lom -y python=3.10
conda activate lom

# Install PyTorch with CUDA support
conda install pytorch==2.4.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# Alternative for RTX 5090 users: install pytorch by following way
# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install pip and dependencies
python -m pip install pip==21.3
pip install -r requirements.txt

# Install additional packages
pip install turbot5 -U
# Alternative for RTX 5090 users: upgrade triton to support the new architecture
# pip install --upgrade "git+https://github.com/openai/triton.git@main#egg=triton&subdirectory=python"
# export TRITON_JIT_CUDA_ARCHITECTURES=$(
#   python - <<'EOF'
# import torch
# p = torch.cuda.get_device_properties(0)
# print(f"{p.major}{p.minor}")
# EOF
# )

# Install NLP tools
python -m spacy download en_core_web_sm

# Set up fairseq (required for some components)
mkdir -p third_party
cd third_party
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ../..

# Version Conflict
pip install --upgrade "omegaconf>=2.2,<2.4" "hydra-core>=1.3,<1.4"
```

### Setting Up Blender for Rendering

We use [TEMOS](https://github.com/Mathux/TEMOS) for rendering. Install it with our provided script:

```bash
# Execute the setup script to install Blender and its dependencies
chmod +x setup_blender.sh
./setup_blender.sh
```

This script will:
1. Download and extract Blender 2.93.18
2. Verify the Blender Python path
3. Install all necessary Python packages for rendering


</details>

## 📥 Required Resources

<details>

Please register an account on the [Max Planck Institute for Intelligent Systems (MPI-IS) website](https://smpl-x.is.tue.mpg.de/index.html) to access the necessary SMPLX models. Then download the SMPLX models, Hubert, T5, and T2M metrics computation checkpoints by running the following script:

```bash
chmod +x build_resources.sh
./build_resources.sh
```

After running the script, you will have the following directory structure:
```
model_files/
├── hubert_models/     # Hubert audio tokenizer models
├── smplx_models/      # SMPLX body models
├── FLAME2020/         # FLAME face models
├── t2m_evaluators/    # Text-to-Motion evaluation metrics
└── t5_models/         # T5 language models
```
</details>

## 📦 Pretrained Models

<details>

Pretrained models are gradually uploading! Visit the [Hugging Face](https://huggingface.co/JuzeZhang/language_of_motion) repository to download them.

</details>

## 🚀 Quick Start

<details>
<summary><b>Text-to-Motion Generation</b></summary>

```bash
python demo.py --cfg configs/demo_text2motion.yaml --text examples/text2motion.txt --task text2motion --render
```

</details>

<details>
<summary><b>Co-speech Gesture Generation</b></summary>

```bash
python demo.py --cfg configs/demo_cospeech.yaml --audio examples/2_scott_0_111_111.wav --task cospeech --render
```
After running the demo scripts, the generated motion results (including rendered videos and motion data) will be saved in the `./results` directory. For text-to-motion generation, you'll find the motion sequences in `.npz` format and rendered videos in `.mp4` format. For co-speech gesture generation, the results will include synchronized motion and audio in a single video file.

</details>


## 🗃️ Data Preparation

For detailed instructions on data preparation and preprocessing, please refer to the [Datasets Guide](./preprocess/README.md).

## 🔄 Training Pipeline

<details>
<summary><b>1. Compositional Motion Tokenization (VQ-VAE Training)</b></summary>

**📖 [Detailed Documentation](./Compositional_Tokenization.md)**

This stage trains separate VQ-VAE models for different body regions. From our experiments, we found that using 256 codebook dim with a 512 codebook size yields better performance for the face, hands, and upper body, while the lower body performs better with 128 codebook dim and a 512 codebook size. Accordingly, we provide this configuration file for reference.
For detailed training procedures, metrics, and troubleshooting, see the [Compositional Motion Tokenization Guide](./Compositional_Tokenization.md).

**Quick Start Commands:**

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

Global(The compositional tokenizer didn't include any golbal information or global translation, so we still need a global translation predictor, this part are heavliy borrowed from [EMAGE](https://github.com/PantoMatrix/PantoMatrix) ):
```bash
python -m train --cfg configs/config_mixed_stage1_vae_global_wo_mesh_lr1e-4.yaml --nodebug
```

Once we finish the compositional tokenizer training, we will get 5 checkpoints for face, hand, upper, lower and global translation. we can convert the whole BEAT2 and AMASS dataset through following;
```bash
python -m scripts.get_compositional_motion_code --cfg configs/config_mixed_stage1_vq_compositional.yaml
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
> All checkpoints reported here were trained on AMASS and BEAT2 datasets to ensure stronger performance. If you want to reproduce the result shown in [paper](https://arxiv.org/pdf/2412.10523), please use the checkpoint provided from EMAGE that only trained on beat2 speaker2 only, which is only used for metrics computation to gurantee the fairness with other methods. 

The result will be saved at: #TOKENS_DS4#

Here you can use the provided script to compare the originial sequence and the reconstructed motion.
```bash
python -m scripts.inference_compositional_motion_code --cfg configs/config_mixed_stage1_vq_compositional.yaml
```
the npz files and rendering result will be generated within the $data_root$/reconstructed_motion_ds4 path.

Audio Tokenizer, in this work we choose Hubert as our audio tokenizer, while we used the original version of Hubert provided [here](), which is uncommon at this stage. We recommand the newer verision of hubert with higher compatibility。
```bash
python -m scripts.get_speech_code_beat2 --beat2_root "/path/to/your/beat2"
```

```bash
python -m scripts.get_speech_code_librispeech --data_path "/path/to/your/librispeech"
```

</details>

<details>
<summary><b>2. Language Model Pretraining</b></summary>

Pretrained on BEAT2 speaker2 only, used exclusively for fair comparison. This version uses tokenizers and datasets trained only on BEAT2 speaker2:


```bash
 python -m train --cfg configs/config_mixed_stage2_speaker2.yaml --nodebug
```


Normal Version - Can be trained on large scale datasets without numerical comparison constraints:

```bash
 python -m train --cfg configs/config_mixed_stage2.yaml --nodebug
```

</details>

<details>
<summary><b>3. Task-Specific Fine-tuning </b></summary>

Text-to-motion

```bash
 python -m train --cfg configs/config_mixed_stage3_t2m.yaml --nodebug
```

Audio-to-motion

```bash
 python -m train --cfg configs/config_mixed_stage3_a2m.yaml --nodebug
```



Stay tuned for updates on our training procedures and best practices.
</details>


## Evaluation
To evaluate the co-speech metrics, please first update the trained model checkpoint paths in `configs/config_mixed_stage3_a2m.yaml`:

- `TEST.CHECKPOINTS_FACE`
- `TEST.CHECKPOINTS_HAND` 
- `TEST.CHECKPOINTS_UPPER`
- `TEST.CHECKPOINTS_LOWER`

Then, run the following command:

```bash
python -m test --cfg configs/config_mixed_stage3_a2m.yaml
```

**Note:** The demo checkpoint named "Instruct_Mixed_A2M_LM.ckpt" is provided for visualization purposes only. When training your own model, you will observe performance curves similar to those shown below. However, the results presented in the paper do not represent the optimal performance achievable with this framework.





## 📝 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@article{chen2024language,
  title={The Language of Motion: Unifying Verbal and Non-verbal Language of 3D Human Motion},
  author={Chen, Changan and Zhang, Juze and Lakshmikanth, Shrinidhi K and Fang, Yusu and Shao, Ruizhi and Wetzstein, Gordon and Fei-Fei, Li and Adeli, Ehsan},
  journal={CVPR},
  year={2025}
}
```

## Acknowledgements

This project was partially funded by NIH grant R01AG089169 and UST. The authors would also like to thank Georgios Pavlakos for his valuable discussion, Chaitanya Patel, Jingyan Zhang, and Bin Li for their feedback on the paper.

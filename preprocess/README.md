# Data Preprocessing

This directory contains preprocessing scripts for the datasets used in the Language of Motion project.

## Required Datasets

To train the model, you need to download the following datasets:

1. **AMASS**: A comprehensive human motion dataset from [AMASS website](https://amass.is.tue.mpg.de/), with text annotations from [HumanML3D](https://github.com/EricGuo5513/HumanML3D).
2. **BEAT2**: A co-speech gesture dataset featuring synchronized speech, emotion labels, and motion data. Available from the [BEAT website](https://drive.google.com/drive/folders/1ukbifhHc85qWTzspEgvAxCXwn9mK4ifr).
3. **LibriSpeech**: A large-scale corpus of read English speech (1000+ hours). Download from the [LibriSpeech website](https://www.openslr.org/12).

## Dataset Structure

Organize your downloaded datasets according to the following directory structure:

```
datasets/
├── AMASS/
├── BEAT2/
    ├── beat_chinese_v2.0.0/
    ├── beat_english_v2.0.0/
    ├── beat_japanese_v2.0.0/
    ├── beat_spanish_v2.0.0/
└── LibriSpeech/
```


## detail

Before running any preprocessing scripts, please download the required datasets:

### AMASS Dataset
Download the AMASS dataset using our provided script:
```bash
./preprocess/amass_download.sh
```

Make sure you have registered at https://smpl-x.is.tue.mpg.de/ and agreed to the SMPLX license terms before running the download script.

### Process AMASS Dataset
```bash
python preprocess/dataset_process_amass.py
    --smplx_path "/path/to/your/smplx_models"
    --dataset_path_original "/path/to/your/data"
    --dataset_path_processed "/path/to/your/data"
    --index_path "/path/to/your/index.csv"
    --ex_fps 30
```

For example

```bash
python preprocess/dataset_process_amass.py
    --smplx_path ./model_files/smplx_models
    --dataset_path_original /data/datasets/AMASS_original_smplx
    --dataset_path_processed /data/datasets/AMASS_processed
    --index_path ./preprocess/index.csv
    --ex_fps 30
```

### BEAT2 Dataset
The original BEAT2 dataset structure is pretty good. Let's keep using their structure.
And then generate the mean face template used for evaluation metrics(FFD). Note that here we only want to train the facial expression and pose, so during the evaluation, we skip the shape parameter. This step is not common but we don't need to compare with any other methods.

Generate only without global rotation template (what we did)
```bash
python -m preprocess.beat2_face_template_generator --template_type without_global
```
Generate only with global rotation template
```bash
python -m preprocess.beat2_face_template_generator --template_type with_global
```

And also download the emotion label from [BEAT dataset](https://huggingface.co/datasets/H-Liu1997/BEAT)


### LibriSpeech Dataset




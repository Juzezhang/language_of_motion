import torchaudio
from torchaudio.datasets import LIBRISPEECH
import torch
import os
import numpy as np
from os.path import join
import argparse
import joblib
from fairseq import checkpoint_utils

# Check if CUDA is available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up command-line arguments
parser = argparse.ArgumentParser('exp_motion command line tools')
parser.add_argument('--used_set', type=str, default="train-clean-100", help="Specify the dataset split to use")
parser.add_argument('--data_path', type=str, default="/data/data/LibriSpeech", help="Path to the LibriSpeech data")
parser.add_argument('--kmeans_path', type=str, default="checkpoint/hubert_base_ls960_L9_km500.bin", help="Path to the K-means model")
args = parser.parse_args()

used_set = args.used_set
data_path = args.data_path
kmeans_path = args.kmeans_path

# Load LibriSpeech dataset
dataset = LIBRISPEECH(data_path, url=used_set, download=False)

# Load the pre-trained HuBERT model
hubert_checkpoint = 'checkpoint/hubert_base_ls960.pt'
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([hubert_checkpoint])
hubert_model = models[0]
hubert_model.eval()  # Set the model to evaluation mode
hubert_model = hubert_model.to(device)  # Move model to GPU if available

# Load the K-means model for quantization
kmeans_model = joblib.load(kmeans_path)

def extract_features(model, audio):
    """
    Extract features from the audio using the HuBERT model.

    Args:
    - model: The HuBERT model.
    - audio: The input audio array.

    Returns:
    - features: Extracted features from the audio.
    """
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)  # Convert audio to tensor and move to GPU
    with torch.no_grad():
        features = model.extract_features(source=audio_tensor, padding_mask=None)[0]
    return features.squeeze().cpu().numpy()

def quantize_features(features, kmeans_model):
    """
    Quantize the extracted features using the K-means model.

    Args:
    - features: Extracted features from the audio.
    - kmeans_model: The K-means model used for quantization.

    Returns:
    - quantized_indices: The quantized feature indices.
    """
    quantized_indices = kmeans_model.predict(features)
    return quantized_indices

# Define the output directory
output_dir = join(data_path, "Quantized_LibriSpeech_Hubert", used_set)
os.makedirs(output_dir, exist_ok=True)

# Process each audio file in the dataset
for i, (audio, sample_rate, transcript, speaker_id, chapter_id, utterance_id) in enumerate(dataset):
    # Extract and quantize audio features
    features = extract_features(hubert_model, audio.squeeze().numpy())
    quantized_indices = quantize_features(features, kmeans_model)

    # Create the directory structure to save the results
    speaker_dir = os.path.join(output_dir, str(speaker_id))
    chapter_dir = os.path.join(speaker_dir, str(chapter_id))
    os.makedirs(chapter_dir, exist_ok=True)

    # Save the quantized indices
    output_path = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}-{utterance_id:04d}.npy")
    np.save(output_path, quantized_indices)

    if i % 100 == 0:
        print(f"Processed {i + 1}/{len(dataset)} files")
import torchaudio
import torch
import os
import numpy as np
from os.path import join
import argparse
import joblib
from fairseq import checkpoint_utils
from tqdm import tqdm
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser('exp_motion command line tools')
# parser.add_argument('--wav_folder', type=str, required=True, help="Path to the folder containing .wav files")
parser.add_argument('--wav_folder', type=str, default="/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat2_original/beat_v2.0.0/beat_english_v2.0.0/wave16k", help="Path to the folder containing .wav files")
parser.add_argument('--kmeans_path', type=str, default="../../checkpoint/hubert_base_ls960_L9_km500.bin",
                    help="Path to the K-means model")
parser.add_argument('--output_dir', type=str, default="Quantized_Wav_Hubert",
                    help="Directory to save the quantized outputs")
args = parser.parse_args()

wav_folder = args.wav_folder
kmeans_path = args.kmeans_path
output_dir = args.output_dir

# Load HuBERT pre-trained model
hubert_checkpoint = '../../checkpoint/hubert_base_ls960.pt'
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([hubert_checkpoint])
model = models[0]
model.eval()
model = model.to(device)  # Move model to GPU if available

# Load K-means model
kmeans_model = joblib.load(kmeans_path)


# Function to tokenize audio using HuBERT
def tokenize_audio(model, audio):
    audio = torch.tensor(audio).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
    with torch.no_grad():
        # Extract features using HuBERT model
        features = model.extract_features(source=audio, padding_mask=None)[0]
    return features.squeeze().cpu().numpy()


# Function to quantize features
def quantize_features(features, kmeans_model):
    # Quantize the extracted features using K-means model
    quantized_indices = kmeans_model.predict(features)
    return quantized_indices


# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each .wav file in the provided folder
for wav_file in tqdm(os.listdir(wav_folder)):
    if wav_file.endswith(".wav"):
        wav_path = join(wav_folder, wav_file)

        # Load the wav file
        waveform, sample_rate = torchaudio.load(wav_path)

        # Tokenize the audio
        tokenized_audio = tokenize_audio(model, waveform.squeeze().numpy())

        if len(tokenized_audio.shape) < 2 :
            tokenized_audio = tokenized_audio.reshape(1,-1)
        # Quantize the tokenized audio features
        quantized_indices = quantize_features(tokenized_audio, kmeans_model)

        # Save the quantized indices
        output_path = join(output_dir, wav_file.replace(".wav", ".npy"))
        # np.save(output_path, quantized_indices)

        # print(f"Processed and saved tokens for {wav_file}")

print("Processing complete!")
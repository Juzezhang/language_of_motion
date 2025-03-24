import os
import torch
import torchaudio
import numpy as np
from transformers import EncodecModel, AutoProcessor
from torchaudio.datasets import LIBRISPEECH
import argparse
from os.path import join
# Check if CUDA is available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser to handle command-line inputs
parser = argparse.ArgumentParser(description='exp_motion command line tools')
parser.add_argument('--used_set', type=str, default="train-clean-100", help="Specify the dataset split to use")
parser.add_argument('--data_path', type=str, default="/data/data/LibriSpeech", help="Specify the path to the LibriSpeech dataset")
args = parser.parse_args()

# Extract command-line arguments
used_set = args.used_set
data_path = args.data_path

# Set output directory for tokenized data
output_dir = join(data_path, "Quantized_LibriSpeech_Encodec", used_set)
os.makedirs(output_dir, exist_ok=True)

# Load LibriSpeech dataset
dataset = LIBRISPEECH(data_path, url=used_set, download=False)

# Load the pre-trained EnCodec model and processor
model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

model.eval()  # Set the model to evaluation mode
model = model.to(device)  # Move model to GPU if available

# Function to save tokens to a file
def save_tokens(tokens, output_path):
    np.save(output_path, tokens)

# Process each audio sample in the dataset
for i, (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id) in enumerate(dataset):
    # Resample the audio to 24kHz if necessary
    if sample_rate != 24000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
        waveform = resampler(waveform)

    # Preprocess the waveform for the model
    inputs = processor(waveform.squeeze(), sampling_rate=24000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

    # Encode the waveform using the EnCodec model
    with torch.no_grad():
        encoded_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])

    # Extract tokens (compressed audio representation)
    tokens = encoded_outputs["audio_codes"][0,0,0].cpu().numpy()

    # Create the directory structure to save the results
    speaker_dir = os.path.join(output_dir, str(speaker_id))
    chapter_dir = os.path.join(speaker_dir, str(chapter_id))
    os.makedirs(chapter_dir, exist_ok=True)


    # Construct the output file path and save the tokens
    output_path = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}-{utterance_id:04d}.npy")
    save_tokens(tokens, output_path)

    # Print progress every 100 files
    if i % 100 == 0:
        print(f"Processed {i + 1}/{len(dataset)} files")

print("Tokenization complete!")
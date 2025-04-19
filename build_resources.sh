#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Fetch SMPLX data
echo -e "\nBefore you continue, you must register at https://smpl-x.is.tue.mpg.de/ and agree to the SMPLX license terms."
read -p "Username (SMPLX):" username
read -p "Password (SMPLX):" password
username=$(urle $username)
password=$(urle $password)

# Create directories for models
mkdir -p model_files
mkdir -p model_files/smplx_models
mkdir -p model_files/smplx_models/smplx
mkdir -p model_files/t2m_evaluators
mkdir -p model_files/hubert_models
mkdir -p model_files/t5_models

echo -e "\nDownloading SMPLX 2020 model files..."
# Download the neutral SMPLX model
echo "Downloading SMPLX2020 neutral model..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz' -O './model_files/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# Download the SMPLX lockedhead model
echo "Downloading SMPLX lockedhead model..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip' -O './model_files/smplx_models/smplx_lockedhead_20230207.zip' --no-check-certificate --continue

# Extract lockedhead model to a temporary directory
echo "Extracting SMPLX lockedhead model..."
mkdir -p ./model_files/temp_smplx
unzip -o './model_files/smplx_models/smplx_lockedhead_20230207.zip' -d './model_files/temp_smplx/'

# Find and move all NPZ files to the target directory
echo "Moving model files to the target directory..."
find ./model_files/temp_smplx -name "*.npz" -exec mv {} ./model_files/smplx_models/smplx/ \;

# Check specifically for gendered models
if [ -f "./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_FEMALE.npz" ]; then
    mv ./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_FEMALE.npz ./model_files/smplx_models/smplx/
fi

if [ -f "./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_MALE.npz" ]; then
    mv ./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_MALE.npz ./model_files/smplx_models/smplx/
fi

if [ -f "./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_NEUTRAL.npz" ]; then
    mv ./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_NEUTRAL.npz ./model_files/smplx_models/smplx/
fi

# Clean up
echo "Cleaning up..."
rm -rf './model_files/temp_smplx'
rm -rf './model_files/smplx_models/smplx_lockedhead_20230207.zip'

echo -e "\nSMPLX 2020 model setup completed successfully!"
echo "Models are available in: model_files/smplx_models/"

echo -e "\nDownloading t2m evaluators..."

echo "Downloading t2m evaluators to model_files/t2m_evaluators..."
gdown "https://drive.google.com/uc?id=1AYsmEG8I3fAAoraT4vau0GnesWBWyeT8" -O "./model_files/t2m.tar.gz"

# Extract directly to the target directory
echo "Extracting t2m evaluators..."
tar xfzv "./model_files/t2m.tar.gz" -C "./model_files/t2m_evaluators" --strip-components=1

# Clean up
echo "Cleaning up..."
rm "./model_files/t2m.tar.gz"

echo "Download completed! All models and evaluators are stored in the model_files directory."

echo -e "\nDownloading Hubert models..."
echo "Downloading Hubert Base Model..."
wget "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt" -O "./model_files/hubert_models/hubert_base_ls960.pt" --continue

echo "Downloading Hubert Quantizer..."
wget "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin" -O "./model_files/hubert_models/hubert_base_ls960_L9_km500.bin" --continue

echo "Hubert models download completed!"
echo "Models are available in: model_files/hubert_models/"

echo -e "\nDownloading FLAN-T5 model..."
echo "Setting up Git LFS and cloning the model repository..."
git lfs install
cd model_files/t5_models
git clone https://huggingface.co/google/flan-t5-base
cd ../..

echo "FLAN-T5 model download completed!"
echo "Model is available in: model_files/t5_models/flan-t5-base/"
#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Fetch SMPLX data
echo -e "\nBefore you continue, you must register at https://smpl-x.is.tue.mpg.de/ and agree to the SMPLX license terms."
read -p "Username (SMPLX):" username
read -p "Password (SMPLX):" password
username=$(urle $username)
password=$(urle $password)

# Create directories for amass


echo -e "\n Please enter the path to the AMASS dataset."
read -p "AMASS path:" amass_path
mkdir -p $amass_path/AMASS_original_smplx

echo "Downloading amass smplx version..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/ACCAD.tar.bz2' -O "$amass_path/AMASS_original_smplx/ACCAD.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/BMLmovi.tar.bz2' -O "$amass_path/AMASS_original_smplx/BMLmovi.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/BMLrub.tar.bz2' -O "$amass_path/AMASS_original_smplx/BMLrub.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/CMU.tar.bz2' -O "$amass_path/AMASS_original_smplx/CMU.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/CNRS.tar.bz2' -O "$amass_path/AMASS_original_smplx/CNRS.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/gender_specific/mosh_results/DanceDB.tar.bz2' -O "$amass_path/AMASS_original_smplx/DanceDB.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/DFaust.tar.bz2' -O "$amass_path/AMASS_original_smplx/DFaust.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/EKUT.tar.bz2' -O "$amass_path/AMASS_original_smplx/EKUT.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/EyesJapanDataset.tar.bz2' -O "$amass_path/AMASS_original_smplx/EyesJapanDataset.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/GRAB.tar.bz2' -O "$amass_path/AMASS_original_smplx/GRAB.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/HDM05.tar.bz2' -O "$amass_path/AMASS_original_smplx/HDM05.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/gender_specific/mosh_results/HUMAN4D.tar.bz2' -O "$amass_path/AMASS_original_smplx/HUMAN4D.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/HumanEva.tar.bz2' -O "$amass_path/AMASS_original_smplx/HumanEva.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/KIT.tar.bz2' -O "$amass_path/AMASS_original_smplx/KIT.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/MoSh.tar.bz2' -O "$amass_path/AMASS_original_smplx/MoSh.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/PosePrior.tar.bz2' -O "$amass_path/AMASS_original_smplx/PosePrior.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/SFU.tar.bz2' -O "$amass_path/AMASS_original_smplx/SFU.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/SOMA.tar.bz2' -O "$amass_path/AMASS_original_smplx/SOMA.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/SSM.tar.bz2' -O "$amass_path/AMASS_original_smplx/SSM.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/TCDHands.tar.bz2' -O "$amass_path/AMASS_original_smplx/TCDHands.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/TotalCapture.tar.bz2' -O "$amass_path/AMASS_original_smplx/TotalCapture.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/Transitions.tar.bz2' -O "$amass_path/AMASS_original_smplx/Transitions.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplx/neutral/mosh_results/WEIZMANN.tar.bz2' -O "$amass_path/AMASS_original_smplx/WEIZMANN.tar.bz2" --no-check-certificate --continue


# BMLhandball only has smpl-h version.
echo "BMLhandball only has smpl-h version."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/BMLhandball.tar.bz2' -O "$amass_path/AMASS_original_smplx/BMLhandball.tar.bz2" --no-check-certificate --continue


echo "Successfully downloaded amass dataset!"


echo "Extracting all downloaded datasets..."
cd "$amass_path/AMASS_original_smplx"

# Extract all tar.bz2 files
for file in *.tar.bz2; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xjf "$file"
    fi
done

echo "Extraction completed!"
echo "Cleaning up compressed files..."

# Optional: Remove the compressed files after extraction
read -p "Do you want to remove the compressed .tar.bz2 files? (y/n): " remove_compressed
if [[ $remove_compressed == "y" || $remove_compressed == "Y" ]]; then
    rm *.tar.bz2
    echo "Compressed files removed."
else
    echo "Compressed files kept."
fi

cd - > /dev/null  # Return to original directory
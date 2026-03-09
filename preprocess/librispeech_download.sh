#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Create directories for librispeech
echo -e "\n Please enter the path to the librispeech dataset."
read -p "librispeech path:" librispeech_path
mkdir -p $librispeech_path

echo "Downloading librispeech dataset..."
wget https://openslr.trmal.net/resources/12/train-clean-100.tar.gz -O "$librispeech_path/train-clean-100.tar.gz" --no-check-certificate --continue
# wget https://openslr.trmal.net/resources/12/train-clean-360.tar.gz -O "$librispeech_path/train-clean-360.tar.gz" --no-check-certificate --continue
# wget https://openslr.trmal.net/resources/12/train-other-500.tar.gz -O "$librispeech_path/train-other-500.tar.gz" --no-check-certificate --continue
echo "Successfully downloaded librispeech dataset!"

echo "Extracting all downloaded datasets..."
cd "$librispeech_path"

# Extract all tar.gz files
for file in *.tar.gz; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xzf "$file"
    fi
done

echo "Extraction completed!"
echo "Cleaning up compressed files..."

# Optional: Remove the compressed files after extraction
read -p "Do you want to remove the compressed .tar.gz files? (y/n): " remove_compressed
if [[ $remove_compressed == "y" || $remove_compressed == "Y" ]]; then
    rm *.tar.gz
    echo "Compressed files removed."
else
    echo "Compressed files kept."
fi

cd - > /dev/null  # Return to original directory
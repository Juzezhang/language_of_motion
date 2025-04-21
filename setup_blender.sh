#!/bin/bash
# Script to set up Blender for rendering in Language of Motion

# Create third_party directory if it doesn't exist
mkdir -p third_party

# Download Blender
wget 'https://download.blender.org/release/Blender2.93/blender-2.93.18-linux-x64.tar.xz' -O './third_party/blender-2.93.18-linux-x64.tar.xz'

# Extract Blender
tar -xf ./third_party/blender-2.93.18-linux-x64.tar.xz -C ./third_party/

# Double check the blender path
./third_party/blender-2.93.18-linux-x64/blender --background --python-expr "import sys; import os; print('\nThe path to the installation of python of blender can be:'); print('\n'.join(['- '+x.replace('/lib/python', '/bin/python') for x in sys.path if 'python' in (file:=os.path.split(x)[-1]) and not file.endswith('.zip')]))"

# Set up Blender's Python environment
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m ensurepip --upgrade
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install --upgrade pip
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install numpy --target=./third_party/blender-2.93.18-linux-x64/2.93/python/lib/python3.9/site-packages
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install matplotlib --target=./third_party/blender-2.93.18-linux-x64/2.93/python/lib/python3.9/site-packages
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install hydra-core --upgrade --target=./third_party/blender-2.93.18-linux-x64/2.93/python/lib/python3.9/site-packages
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install moviepy==1.0.3 --target=./third_party/blender-2.93.18-linux-x64/2.93/python/lib/python3.9/site-packages
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install shortuuid --target=./third_party/blender-2.93.18-linux-x64/2.93/python/lib/python3.9/site-packages
./third_party/blender-2.93.18-linux-x64/2.93/python/bin/python3.9 -m pip install -r preprocess/requirements_render.txt --target=./third_party/blender-2.93.18-linux-x64/2.93/python/lib/python3.9/site-packages

echo "Blender setup completed successfully!" 
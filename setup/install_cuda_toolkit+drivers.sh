#!/bin/bash

lspci | grep -i nvidia # Verify You Have a CUDA-Capable GPU
hostnamectl # Verify You Have a Supported Version of Linux
gcc --version # Verify You Have a Supported Version of GCC

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

sudo apt-get install -y cuda-drivers
sudo apt-get -y install cudnn9-cuda-12

export PATH=${PATH}:/usr/local/cuda-12.9/bin
sudo apt-get autoremove -y
sudo apt-get clean -y
sudo apt-get autoclean -y

sudo reboot

# Optional: Move cuda to /data
# sudo mv /usr/local/cuda-12.9 /data/cuda-12.9
# sudo ln -s /data/cuda-12.9 /usr/local/cuda-12.9
nvcc --version
nvidia-smi
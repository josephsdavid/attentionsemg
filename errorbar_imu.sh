#!/usr/bin/env bash
#SBATCH -J error_bar
#SBATCH -e result/imeb-%j.txt
#SBATCH -o result/imeb-%j.txt
#SBATCH -p gpgpu-1  --gres=gpu:1 --mem=96G
#SBATCH -t 1440
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python imu_error_bar.py "$@"

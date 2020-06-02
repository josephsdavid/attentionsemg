#!/usr/bin/env bash
#SBATCH -J cv
#SBATCH -e result/%j.txt
#SBATCH -o result/%j.txt
#SBATCH -p gpgpu-1  --gres=gpu:1 --mem=180G
#SBATCH -t 1440
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python error_bar.py "$@"

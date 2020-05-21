#!/usr/bin/env bash
#SBATCH -J error_bar
#SBATCH -e result/eb-%j.txt
#SBATCH -o result/eb-%j.txt
#SBATCH -p v100x8  --gres=gpu:1 --mem=96G
#SBATCH -t 1440
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python imu_error_bar.py "$@"

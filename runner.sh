#!/usr/bin/env bash
#SBATCH -J nina4
#SBATCH -e result/17class.txt
#SBATCH -o result/17class.txt
#SBATCH -p gpgpu-1  --gres=gpu:1 --mem=90G
#SBATCH -t 10080
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python noimu_17.py

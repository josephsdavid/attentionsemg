#!/usr/bin/env bash
#SBATCH -J ablation
#SBATCH -e result/ab3-%j.txt
#SBATCH -o result/ab3-%j.txt
#SBATCH -p v100x8  --gres=gpu:1 --mem=96G
#SBATCH -t 10080
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python rnn_check.py

#!/usr/bin/env bash
#SBATCH -J ablation
#SBATCH -e result/ab1-%j.txt
#SBATCH -o result/ab1-%j.txt
#SBATCH -p v100x8  --gres=gpu:1 --mem=96G``
#SBATCH -t 10080
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python ablation_stage_1.py

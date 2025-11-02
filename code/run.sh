#!/bin/bash
#SBATCH --partition=pool1           
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --output=run1_%A.out
#SBATCH --error=run1_%A.err
#SBATCH --job-name=coalescent

source ~/miniforge3/bin/activate

conda activate thesis-coalescent

bash "/home/ssabata/Spatial-Dynamics-of-the-Coalescence/code/pipeline.sh" --spatial-only

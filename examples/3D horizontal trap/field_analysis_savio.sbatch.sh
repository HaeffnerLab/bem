#!/bin/bash
#SBATCH --job-name=bem_simulation_3D_horizontal
#SBATCH --account=ozgursahin    # e.g. ac_abc123
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load Anaconda
module purge
module load anaconda3

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"

# Activate your environment (replace with yours)
conda activate my_env

# Run your python script (replace with yours)
python my_script.py

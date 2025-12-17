#!/bin/bash
#SBATCH --job-name=bem_simulation_3D_horizontal
#SBATCH --account=ozgursahin
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load Anaconda
module purge
module load anaconda3

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"

# Activate your environment (replace with yours)
conda activate bem39

# Run your python script (replace with yours)
python -u II_Field_Simulation.py

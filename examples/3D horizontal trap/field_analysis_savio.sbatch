#!/bin/bash
#SBATCH --job-name=bem_simulation_3D_horizontal
#SBATCH --account=fc_haeffnerbem
#SBATCH --partition=savio3_xlmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1024G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load Anaconda
module purge
module load anaconda3

# Initialize conda for non-interactive shells
eval "$(conda shell.bash hook)"

# Activate your environment (replace with yours)

conda activate bem39

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1


# Run your python script (replace with yours)
srun python -u II_Field_Simulation.py

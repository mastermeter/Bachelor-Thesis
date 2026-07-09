#!/bin/bash
#SBATCH --job-name=congeo_vis    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --output=logs/job%j.out
#SBATCH --error=logs/job%j.err
#SBATCH --partition=Dance
#SBATCH --nodelist=disco
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

if [ -z "$USER_NAME" ]; then
    echo "Must specify user.name"
    echo "Exemple : sbatch --export=ALL,USER_NAME=alexandre.venturi train.sh"
    exit 1
fi

cd "/home/${USER_NAME}/Bachelor-Thesis/ConGeo"

mkdir -p logs

apptainer exec \
    --nv \
    --bind "/data/space/datasets/${USER_NAME}/VIGOR:/mnt/vigor" \
    ./version2.sif \
    python3 visualize_topk.py
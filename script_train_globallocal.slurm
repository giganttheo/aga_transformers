#!/bin/sh
#SBATCH -J t5-graph-base-16k
#SBATCH -n 1
#SBATCH --nnodes=1
#SBATCH --nodelist=n[1-5]
#SBATCH -p all
#SBATCH -t 24:00:00
#SBATCH -o ./%j.out
#SBATCH -e ./%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=theo.gigant@l2s.centralesupelec.fr
#launch train
bash train_globallocal.sh
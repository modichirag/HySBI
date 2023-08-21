#!/bin/bash
#SBATCH -p ccm
#SBATCH -C skylake
#SBATCH --time=1:00:00
#SBATCH -N 1
#SBATCH --job-name=pksub


# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load gcc/7.5.0 openmpi/1.10.7
source activate ptorch

i0=$1
i1=$2
echo $i0 $i1

time python -u pk_sub.py $i0 $i1

#!/bin/bash
#SBATCH -p ccm
#SBATCH -C skylake
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --job-name=sp_snle
#SBATCH -o ../logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load gcc/7.5.0 openmpi/1.10.7
source activate ptorch

i0=$1
i1=$(($i0+200))

# time python -u sampler_pt.py $i0


echo $i0 $i1
for((i=${i0} ; i<=${i1} ; i+=1))
do
    echo $i    
    time python -u sample_sweep.py $i  
done    

# wait

    

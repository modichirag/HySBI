#!/bin/bash
#SBATCH -p gpu
#SBATCH -C a100-40gb
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --job-name=pt
#SBATCH -o ../logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load gcc/7.5.0 openmpi/1.10.7
module load cuda cudnn
source activate ptorch
cd ..

i0=$1
i1=$(($i0+500))

# time python -u sampler_pt.py $i0


echo $i0 $i1
for((i=${i0} ; i<=${i1} ; i+=1))
do
    echo $i    
    time python -u sampler_pt.py $i  
done    

# wait

    

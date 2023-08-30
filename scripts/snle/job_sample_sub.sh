#!/bin/bash
#SBATCH -p ccm
#SBATCH -C skylake
#SBATCH --time=10:00:00
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
i1=$2
echo $i0 $i1
summary="wavelets"

#cfgfolder="kmax0.5-kmin0.001-logit-standardize"
#cfgfolder="dk1-kmax0.5-kmin0.15-logit-meanf-nsubs8-splits2-standardize"
#cfgfolder="dk2-kmax0.5-kmin0.15-logit-meanf-nsubs8-splits2-standardize"
cfgfolder="J2-M4-Q4-e0.1-kc0.67-logit-meanf-nsubs8-splits2-standardize"

for((i=${i0} ; i<=${i1} ; i+=1))
do
    echo $i    
    time python -u sample_sweep.py --isim $i  --testsims --cfgfolder $cfgfolder --subdata --summary $summary
done    

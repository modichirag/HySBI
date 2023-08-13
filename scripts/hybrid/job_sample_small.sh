#!/bin/bash
#SBATCH -p ccm
#SBATCH -C skylake
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --job-name=sp_hybrid
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

cfgfolder="kmax0.5-kmin0.001-ksplit0.15-nsubs1-splits1-standardize-standardize_cond"
#cfgfolder="kmax0.5-kmin0.001-ksplit0.1-nsubs1-splits1-standardize-standardize_cond"

#cfgfolder="kmax0.5-kmin0.001-ksplit0.15-meanf-nsubs8-splits2-standardize-standardize_cond"
#cfgfolder="kmax0.5-kmin0.001-ksplit0.1-meanf-nsubs8-splits2-standardize-standardize_cond"

# time python -u sample_sweep.py  --isim $i0  --testsims --cfgfolder $cfgfolder
for((i=${i0} ; i<=${i1} ; i+=1))
do
    echo $i    
    time python -u sample_small.py --isim $i  --testsims --cfgfolder $cfgfolder
done    

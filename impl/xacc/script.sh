#!/bin/bash
#SBATCH --comment "Hello ROMEO!"
#SBATCH -J "TEST 1"

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --time=00:30:00

#SBATCH -n 4
#SBATCH -N 4

#SBATCH --gres=gpu:1

#module load oscar-modules/1.0.3 java/1.8.0_102 impi/5.1.1 intel/2016 cuda/7.5
echo $CUDA_VISIBLE_DEVICES

mpirun -np 4 ./a.out  


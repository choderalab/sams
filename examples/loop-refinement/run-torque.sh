#!/bin/bash
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=72:00:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q gpu
#
#PBS 
# nodes: number of nodes
#   ppn: how many cores per node to use
#PBS -l nodes=1:ppn=1:gpus=1:gtxtitan
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N alchemical-loop

if [ -n "$PBS_O_WORKDIR" ]; then 
    cd $PBS_O_WORKDIR
fi

cat $PBS_GPUFILE
nvidia-smi 

# Run the simulation with verbose output:
echo "Running alchemical loop refinement..."
python AlchemicalLoopSoftening.py
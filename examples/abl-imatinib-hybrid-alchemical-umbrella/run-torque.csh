#!/bin/tcsh
#  Batch script for mpirun job on cbio cluster.
#
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=24:00:00
#
# join stdout and stderr
#PBS -j oe
#
# spool output immediately
#PBS -k oe
#
# specify queue
#PBS -q cpath
#
# nodes: number of 8-core nodes
#   ppn: how many cores per node to use (1 through 8)
#       (you are always charged for the entire node)
#PBS -l nodes=1:ppn=1:gpus=1:shared:gtxtitanpas
#
# export all my environment variables to the job
##PBS -V
#
# job name (default = name of script file)
#PBS -N sams
#
# specify email
#PBS -M jchodera@gmail.com
#
# mail settings
#PBS -m n

cd $PBS_O_WORKDIR

echo | grep PYTHONPATH
which python

cat $PBS_GPUFILE

# Only use one OpenMM CPU thread.
setenv OPENMM_CPU_THREADS 1

date
python hybrid-alchemical-umbrella2.py
date


#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH -J "BSPMM"
#SBATCH -o "%j-bspmm.out"
#SBATCH -t 120:00
#SBATCH -p development
#SBATCH --mail-user=michalowicz.2@osu.edu
#SBATCH --mail-type=ALL



source ~/.bashrc
exp $HOME/BSPMM/binaries
#module load gcc/12.2.0
hostfile=$PWD/hostfile-$SLURM_JOBID
envs="MV2_HOMOGENEOUS_CLUSTER=1"

for ppn in 1 2 4 8;
do
    proc_grid=0
    if [[ $ppn == 1 ]]; then
        proc_grid=2
    elif [[ $ppn == 2 ]]; then
        proc_grid=2
    elif [[ $ppn == 4 ]]; then
        proc_grid=3
    elif [[ $ppn == 8 ]]; then
        proc_grid=4;
    fi
    
    echo "hostfile and ppn:"
    hostfile=$PWD/hostfile-$ppn
    cat $hostfile
    export SLURM_NNODES=2
    
    set -x
    echo "MPI (use of OpenMPI :/)"
    for mesh in 4 8 16 32; do
        echo "Mesh size = $mesh x $mesh x $mesh"
        for q in `seq 1 4`; do
            mpirun -np $(($SLURM_NNODES * $ppn)) --hostfile $hostfile \
                ./original $mesh $mesh $proc_grid 2 $ppn | grep "TOTAL TIME"
        done
    done

    if [[ $ppn > 1 ]]; then
        for q in `seq 1 4`; do
            echo "Mesh size = 64 x 64 x 64"
            mpirun -np $(($SLURM_NNODES * $ppn)) --hostfile $hostfile \
                $envs ./original 64 64 $proc_grid 2 $ppn | grep "TOTAL TIME"
        done
    fi



    echo "blocking OSHMEM"
    for mesh in 4 8 16 32; do
        echo "Mesh size = $mesh x $mesh x $mesh"
        for q in `seq 1 4`; do
            mpirun -np $(($SLURM_NNODES * $ppn)) --hostfile $hostfile \
                ./osh_original $mesh $mesh $proc_grid 2 $ppn
        done
    done

    if [[ $ppn > 1 ]]; then
        for q in `seq 1 4`; do
            mpirun -np $(($SLURM_NNODES * $ppn)) --hostfile $hostfile \
                $envs ./osh_original 64 64 $proc_grid 2 $ppn
        done
    fi


    echo "NON-blocking OSHMEM"
    for mesh in 4 8 16 32; do
        echo "Mesh size = $mesh x $mesh x $mesh"
        for q in `seq 1 4`; do
            mpirun -np $(($SLURM_NNODES * $ppn)) --hostfile $hostfile \
                ./osh_nb $mesh $mesh $proc_grid 2 $ppn
        done
    done

    if [[ $ppn > 1 ]]; then
        for q in `seq 1 4`; do
            echo "Mesh size = 64 x 64 x 64"
            mpirun -np $(($SLURM_NNODES * $ppn)) --hostfile $hostfile \
                $envs ./osh_nb 64 64 $proc_grid 2 $ppn
        done
    fi
    set +x
done

rm $hostfile


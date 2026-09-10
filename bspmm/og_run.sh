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
exp $HOME/BSPMM-2/2.3.x/install
exp $HOME/BSPMM-2/oshmpi/install
#module load gcc/12.2.0
#hostfile=$PWD/hostfile-$SLURM_JOBID
envs="MV2_SMP_USE_CMA=0"

hostfile=$PWD/hostfile-2.3.x
configfile=$PWD/configfile-2.3.x

for ppn in 1 2 4 8;
do
    rm $hostfile
    touch $hostfile
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
    for i in 10.1.1.82 10.1.1.210; do
        echo "$i:$ppn" >> $hostfile
    done

    
    
    echo "hostfile and ppn:"
    cat $hostfile
    export SLURM_NNODES=2
    nprocs=$(($SLURM_NNODES * $ppn))

            
    
      #  echo "MPI MVAPICH-time!!!"
    for mesh in 4 8 16 32; do
        echo "Mesh size = $mesh x $mesh x $mesh"
        rm -f $configfile
        touch $configfile
        for i in 1 2; do
            echo "-n $ppn : $PWD/original $mesh $mesh $proc_grid 2 $ppn" >> \
                $configfile
        done

        for q in `seq 1 5`; do
          #  set -x
            mpirun_rsh --hostfile $hostfile --config $configfile $envs \
                2>/dev/null | grep -C2 "TOTAL TIME" | tail -n2
                 
          #  set +x
            #exit 2
            #-np $(($SLURM_NNODES * $ppn)) --hostfile $hostfile \
            #    ./original $mesh $mesh $proc_grid 2 $ppn | grep "TOTAL TIME"
        done
    done
done

rm $hostfile
rm $configfile


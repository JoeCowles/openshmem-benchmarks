#!/bin/bash
#SBATCH --job-name=stencil8192
#SBATCH --partition=zen4
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=stencil8192_%j.log
set -uo pipefail
cd /mnt/DISCL/home/jcowles/MLIR_testing/stencil_mlir
SOS_DIR=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/openshmem-runtime/SOS-v1.5.2
export PATH="${SOS_DIR}/bin:${PATH}"; export LD_LIBRARY_PATH="${SOS_DIR}/lib:${LD_LIBRARY_PATH:-}"
# small heap (stencil uses static symmetric data) + explicit --mem: the partition
# default is DefMemPerCPU=6045MB * 1 CPU/task ~= 6GB, so a big heap OOM-kills.
export SHMEM_SYMMETRIC_HEAP_SIZE=1000000000; export SHMEM_OFI_PROVIDER="verbs;ofi_rxm"
export SHMEM_CMA_GET_MAX=0 SHMEM_CMA_PUT_MAX=0
export OSHRUN_LAUNCHER="srun --mpi=pmi2 --nodes=4 --ntasks-per-node=1 --mem=32G"
gettime(){ grep -iE "Elapsed" | awk '{for(i=1;i<=NF;i++){if($i=="second(s),")s=$(i-1); if($i=="milliseconds,")m=$(i-1); if($i=="micro")u=$(i-1)} print s+m/1000.0+u/1000000.0}'; }
ct=$(timeout 300 oshrun -n 4 setarch "$(uname -m)" -R build-ctrl-8192/stencil 2>/dev/null | gettime)
at=$(timeout 300 oshrun -n 4 setarch "$(uname -m)" -R build-agg-8192/stencil 2>/dev/null | gettime)
sp=$(python3 -c "print(f'{$ct/$at:.1f}')" 2>/dev/null || echo "-")
echo "4,8192,$ct,$at,$sp" >> /mnt/DISCL/home/jcowles/MLIR_testing/paper/stencil_ncols_sweep.csv
echo "stencil NCOLS=8192 4PE: ctrl=$ct agg=$at speedup=${sp}x"

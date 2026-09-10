#!/bin/bash
# build_blk.sh — compile halo_blk.c through the ClangIR->OpenSHMEM-MLIR->LLVM
# pipeline, toggling the block message-aggregation pass.
#   PASS=off  -> baseline (naive block puts)
#   PASS=blk  -> --openshmem-scalar-to-bulk (block coalescing)
#   DEFS="GATHERED NBLK=4096" etc
set -e
CLANG=/mnt/DISCL/home/jcowles/MLIR_testing/clangir/build-new/bin/clang
CIR_OPT=/mnt/DISCL/home/jcowles/MLIR_testing/clangir/build-new/bin/cir-opt
SHMEM_OPT=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/build-incubator/bin/shmem-mlir-opt
MLIR_TRANSLATE=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/clangir/build-main/bin/mlir-translate
LLC=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/clangir/build-main/bin/llc
SOS=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/openshmem-runtime/SOS-v1.5.2
OSHCC=${SOS}/bin/oshcc
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC="${SRC:-halo_blk}"
PASS="${PASS:-off}"
BUILD="${HERE}/build-${SRC}-${PASS}"
mkdir -p "$BUILD"
DEFARR=(); for d in ${DEFS:-}; do DEFARR+=(-D"$d"); done
OPT=(); [[ "$PASS" == "blk" ]] && OPT+=("--openshmem-scalar-to-bulk${STB_FLAGS:-}")

echo "=== build $SRC PASS=$PASS DEFS='${DEFS:-}' ==="
$CLANG -std=c99 -O0 -fclangir -emit-cir "${DEFARR[@]}" -I"$SOS/include" "$HERE/src/$SRC.c" -o "$BUILD/$SRC.cir"
$CIR_OPT --cir-goto-solver "$BUILD/$SRC.cir" -o "$BUILD/$SRC.flat.cir"
$SHMEM_OPT --convert-cir-to-openshmem --openshmem-inline-comm-helpers "${OPT[@]}" \
  --cir-flatten-cfg --cir-abi-lowering --convert-openshmem-to-llvm \
  --allow-unregistered-dialect --cir-to-llvm --reconcile-unrealized-casts \
  --mlir-print-op-generic "$BUILD/$SRC.flat.cir" -o "$BUILD/$SRC.llvm.mlir"
$MLIR_TRANSLATE --allow-unregistered-dialect --mlir-to-llvmir "$BUILD/$SRC.llvm.mlir" -o "$BUILD/$SRC.ll"
$LLC -O3 -filetype=obj -relocation-model=pic "$BUILD/$SRC.ll" -o "$BUILD/$SRC.o"
SHMEM_CC="$CLANG" $OSHCC "$BUILD/$SRC.o" -lm -lrt -o "$BUILD/$SRC"
echo "  bin: $BUILD/$SRC   agg_buf=$(grep -c shmem_agg_buf "$BUILD/$SRC.ll" 2>/dev/null || echo 0)"

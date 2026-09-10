#!/bin/bash
# build_stencil.sh — compile stencil.c through ClangIR -> OpenSHMEM-MLIR -> LLVM,
# same pipeline as build_ssca1.noflat.sh. Toggle aggregation with USE_SCALAR_TO_BULK.
#   USE_SCALAR_TO_BULK=0                         -> control (naive per-element)
#   USE_SCALAR_TO_BULK=1 STB_FLAG="=..."         -> compiler aggregation
#   NCOLS=.. LROWS=.. NSTEPS=..                  -> grid/step overrides (-D)
#   BUILD_DIR=..                                 -> output dir (default build-ctrl)
set -e
CLANG=/mnt/DISCL/home/jcowles/MLIR_testing/clangir/build-new/bin/clang
CIR_OPT=/mnt/DISCL/home/jcowles/MLIR_testing/clangir/build-new/bin/cir-opt
SHMEM_OPT=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/build-incubator/bin/shmem-mlir-opt
MLIR_TRANSLATE=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/clangir/build-main/bin/mlir-translate
LLC=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/clangir/build-main/bin/llc
SOS_DIR=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/openshmem-runtime/SOS-v1.5.2
OSHCC=${SOS_DIR}/bin/oshcc
SHMEM_INC=${SOS_DIR}/include
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BUILD_DIR="${BUILD_DIR:-${HERE}/build-ctrl}"
SRC="${HERE}/stencil.c"
mkdir -p "${BUILD_DIR}"
BIN="${BUILD_DIR}/stencil"

DEFS=()
[[ -n "${NCOLS:-}"  ]] && DEFS+=(-DNCOLS=${NCOLS})
[[ -n "${LROWS:-}"  ]] && DEFS+=(-DLROWS=${LROWS})
[[ -n "${NSTEPS:-}" ]] && DEFS+=(-DNSTEPS=${NSTEPS})

STB=()
if [[ "${USE_SCALAR_TO_BULK:-0}" == "1" ]]; then
  STB+=("--openshmem-scalar-to-bulk${STB_FLAG:-}")
fi

echo "=== stencil build (BUILD_DIR=${BUILD_DIR}, STB=${STB[*]:-none}, DEFS=${DEFS[*]:-default}) ==="
${CLANG} -std=c99 -O0 -fclangir -emit-cir "${DEFS[@]}" -I"${SHMEM_INC}" "${SRC}" -o "${BUILD_DIR}/stencil.cir"
${CIR_OPT} --cir-goto-solver "${BUILD_DIR}/stencil.cir" -o "${BUILD_DIR}/stencil.flat.cir"
${SHMEM_OPT} \
  --convert-cir-to-openshmem \
  --openshmem-inline-comm-helpers \
  "${STB[@]}" \
  --cir-flatten-cfg \
  --cir-abi-lowering \
  --convert-openshmem-to-llvm \
  --allow-unregistered-dialect \
  --cir-to-llvm \
  --reconcile-unrealized-casts \
  --mlir-print-op-generic \
  "${BUILD_DIR}/stencil.flat.cir" -o "${BUILD_DIR}/stencil.llvm.mlir"
${MLIR_TRANSLATE} --allow-unregistered-dialect --mlir-to-llvmir "${BUILD_DIR}/stencil.llvm.mlir" -o "${BUILD_DIR}/stencil.ll"
${LLC} -O3 -filetype=obj -relocation-model=pic "${BUILD_DIR}/stencil.ll" -o "${BUILD_DIR}/stencil.o"
SHMEM_CC="${CLANG}" ${OSHCC} "${BUILD_DIR}/stencil.o" -lm -lrt -o "${BIN}"
echo "  Generated: ${BIN}"
echo "  agg buffers: $(grep -c 'shmem_agg_buf' "${BUILD_DIR}/stencil.ll" 2>/dev/null || echo 0)   stacksave: $(grep -c 'stacksave' "${BUILD_DIR}/stencil.ll" 2>/dev/null || echo 0)"

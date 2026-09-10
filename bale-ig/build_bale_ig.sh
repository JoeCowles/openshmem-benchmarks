#!/bin/bash
# build_bale_ig.sh — compile bale_ig.c through ClangIR -> OpenSHMEM-MLIR
# -> LLVM, same pipeline as build_stencil.sh.
#   USE_MANUAL_AGG=1                              -> hand-aggregated oracle (Bulk method)
#   USE_INSPECTOR_EXECUTOR=1 [IE_FLAG="=..."]     -> compiler inspector-executor pass
#   (neither)                                    -> naive control
#   L_TBL_SIZE=.. L_NUM_REQ=.. REPS=..                 -> problem-size overrides (-D)
#   BUILD_DIR=..                                 -> output dir
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
SRC="${HERE}/bale_ig.c"
mkdir -p "${BUILD_DIR}"
BIN="${BUILD_DIR}/bale_ig"

DEFS=()
[[ -n "${L_TBL_SIZE:-}" ]] && DEFS+=(-DL_TBL_SIZE=${L_TBL_SIZE})
[[ -n "${L_NUM_REQ:-}"      ]] && DEFS+=(-DL_NUM_REQ=${L_NUM_REQ})
[[ -n "${REPS:-}"      ]] && DEFS+=(-DREPS=${REPS})
[[ "${USE_MANUAL_AGG:-0}" == "1" ]] && DEFS+=(-DUSE_MANUAL_AGG)

IE=()
if [[ "${USE_INSPECTOR_EXECUTOR:-0}" == "1" ]]; then
  IE+=("--openshmem-inspector-executor${IE_FLAG:-}")
fi

echo "=== bale_ig build (BUILD_DIR=${BUILD_DIR}, IE=${IE[*]:-none}, DEFS=${DEFS[*]:-default}) ==="
${CLANG} -std=c99 -O0 -fclangir -emit-cir "${DEFS[@]}" -I"${SHMEM_INC}" "${SRC}" -o "${BUILD_DIR}/bale_ig.cir"
${CIR_OPT} --cir-goto-solver "${BUILD_DIR}/bale_ig.cir" -o "${BUILD_DIR}/bale_ig.flat.cir"
${SHMEM_OPT} \
  --convert-cir-to-openshmem \
  --openshmem-inline-comm-helpers \
  "${IE[@]}" \
  --cir-flatten-cfg \
  --cir-abi-lowering \
  --convert-openshmem-to-llvm \
  --allow-unregistered-dialect \
  --cir-to-llvm \
  --reconcile-unrealized-casts \
  --mlir-print-op-generic \
  "${BUILD_DIR}/bale_ig.flat.cir" -o "${BUILD_DIR}/bale_ig.llvm.mlir"
${MLIR_TRANSLATE} --allow-unregistered-dialect --mlir-to-llvmir "${BUILD_DIR}/bale_ig.llvm.mlir" -o "${BUILD_DIR}/bale_ig.ll"
${LLC} -O3 -filetype=obj -relocation-model=pic "${BUILD_DIR}/bale_ig.ll" -o "${BUILD_DIR}/bale_ig.o"
SHMEM_CC="${CLANG}" ${OSHCC} "${BUILD_DIR}/bale_ig.o" -lm -lrt -o "${BIN}"
echo "  Generated: ${BIN}"
echo "  ie buffers: $(grep -c 'shmem_ie_mirror\|shmem_ie_buf' "${BUILD_DIR}/bale_ig.ll" 2>/dev/null || echo 0)"

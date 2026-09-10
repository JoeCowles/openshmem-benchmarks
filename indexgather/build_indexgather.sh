#!/bin/bash
# build_indexgather.sh — compile indexgather.c through ClangIR -> OpenSHMEM-MLIR
# -> LLVM, same pipeline as build_stencil.sh.
#   USE_MANUAL_AGG=1                              -> hand-aggregated oracle (Bulk method)
#   USE_INSPECTOR_EXECUTOR=1 [IE_FLAG="=..."]     -> compiler inspector-executor pass
#   (neither)                                    -> naive control
#   LOCAL_LEN=.. NREQ=.. REPS=..                 -> problem-size overrides (-D)
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
SRC="${HERE}/indexgather.c"
mkdir -p "${BUILD_DIR}"
BIN="${BUILD_DIR}/indexgather"

DEFS=()
[[ -n "${LOCAL_LEN:-}" ]] && DEFS+=(-DLOCAL_LEN=${LOCAL_LEN})
[[ -n "${NREQ:-}"      ]] && DEFS+=(-DNREQ=${NREQ})
[[ -n "${REPS:-}"      ]] && DEFS+=(-DREPS=${REPS})
[[ "${USE_MANUAL_AGG:-0}" == "1" ]] && DEFS+=(-DUSE_MANUAL_AGG)

IE=()
if [[ "${USE_INSPECTOR_EXECUTOR:-0}" == "1" ]]; then
  IE+=("--openshmem-inspector-executor${IE_FLAG:-}")
fi

echo "=== indexgather build (BUILD_DIR=${BUILD_DIR}, IE=${IE[*]:-none}, DEFS=${DEFS[*]:-default}) ==="
${CLANG} -std=c99 -O0 -fclangir -emit-cir "${DEFS[@]}" -I"${SHMEM_INC}" "${SRC}" -o "${BUILD_DIR}/indexgather.cir"
${CIR_OPT} --cir-goto-solver "${BUILD_DIR}/indexgather.cir" -o "${BUILD_DIR}/indexgather.flat.cir"
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
  "${BUILD_DIR}/indexgather.flat.cir" -o "${BUILD_DIR}/indexgather.llvm.mlir"
${MLIR_TRANSLATE} --allow-unregistered-dialect --mlir-to-llvmir "${BUILD_DIR}/indexgather.llvm.mlir" -o "${BUILD_DIR}/indexgather.ll"
${LLC} -O3 -filetype=obj -relocation-model=pic "${BUILD_DIR}/indexgather.ll" -o "${BUILD_DIR}/indexgather.o"
SHMEM_CC="${CLANG}" ${OSHCC} "${BUILD_DIR}/indexgather.o" -lm -lrt -o "${BIN}"
echo "  Generated: ${BIN}"
echo "  ie buffers: $(grep -c 'shmem_ie_mirror\|shmem_ie_buf' "${BUILD_DIR}/indexgather.ll" 2>/dev/null || echo 0)"

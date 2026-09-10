#!/bin/bash
# build_prk.sh — compile a PRK SHMEM source through the ClangIR -> OpenSHMEM-MLIR
# -> LLVM pipeline, with the OpenSHMEM optimization passes toggleable so we can
# compare aggregation OFF vs ON on the SAME source (pass-off is the baseline).
#
# Usage:
#   SRC=prk_stencil PASS=off ./build_prk.sh
#   SRC=prk_stencil PASS=stb ./build_prk.sh            # scalar-to-bulk
#   SRC=prk_p2p     PASS=agg AGG_FLAGS="--enable-put" ./build_prk.sh
#
# Env:
#   SRC        base name in src/ (prk_stencil | prk_p2p)     [required]
#   PASS       off | stb | agg                               [default off]
#   STB_FLAGS  extra flags appended to --openshmem-scalar-to-bulk (e.g. "=assume-independent")
#   AGG_FLAGS  extra flags for --openshmem-message-aggregation (e.g. "--enable-put --enable-get")
#   DEFS       extra -D preprocessor defines
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

SRC="${SRC:?set SRC to a base name in src/ (e.g. prk_stencil)}"
PASS="${PASS:-off}"
SRC_C="${HERE}/src/${SRC}.c"
BUILD_DIR="${HERE}/build-${SRC}-${PASS}"
mkdir -p "${BUILD_DIR}"
BIN="${BUILD_DIR}/${SRC}"

# Select optimization passes.
OPT=()
case "${PASS}" in
  off) ;;                                                            # baseline: no aggregation
  stb) OPT+=("--openshmem-scalar-to-bulk${STB_FLAGS:-}") ;;
  agg) OPT+=("--openshmem-message-aggregation" ${AGG_FLAGS:-}) ;;
  *) echo "unknown PASS=${PASS}"; exit 2 ;;
esac

DEFARR=()
for d in ${DEFS:-}; do DEFARR+=(-D"$d"); done

echo "=== build ${SRC} (PASS=${PASS}, OPT=${OPT[*]:-none}) ==="
${CLANG} -std=c99 -O0 -fclangir -emit-cir "${DEFARR[@]}" \
    -I"${HERE}/include" -I"${SHMEM_INC}" "${SRC_C}" -o "${BUILD_DIR}/${SRC}.cir"
${CIR_OPT} --cir-goto-solver "${BUILD_DIR}/${SRC}.cir" -o "${BUILD_DIR}/${SRC}.flat.cir"
${SHMEM_OPT} \
  --convert-cir-to-openshmem \
  --openshmem-inline-comm-helpers \
  "${OPT[@]}" \
  --cir-flatten-cfg \
  --cir-abi-lowering \
  --convert-openshmem-to-llvm \
  --allow-unregistered-dialect \
  --cir-to-llvm \
  --reconcile-unrealized-casts \
  --mlir-print-op-generic \
  "${BUILD_DIR}/${SRC}.flat.cir" -o "${BUILD_DIR}/${SRC}.llvm.mlir"
${MLIR_TRANSLATE} --allow-unregistered-dialect --mlir-to-llvmir "${BUILD_DIR}/${SRC}.llvm.mlir" -o "${BUILD_DIR}/${SRC}.ll"
${LLC} -O3 -filetype=obj -relocation-model=pic "${BUILD_DIR}/${SRC}.ll" -o "${BUILD_DIR}/${SRC}.o"
# PRK helper functions (bail_out, wtime) compiled normally (they don't need the
# aggregation pass); link them alongside the pipeline-compiled kernel object.
HELPERS=()
for h in SHMEM_bail_out wtime; do
  if [[ -f "${HERE}/common/${h}.c" ]]; then
    ${OSHCC} -O3 -std=c99 -I"${HERE}/include" -I"${SHMEM_INC}" -c "${HERE}/common/${h}.c" -o "${BUILD_DIR}/${h}.o" 2>/dev/null
    HELPERS+=("${BUILD_DIR}/${h}.o")
  fi
done
SHMEM_CC="${CLANG}" ${OSHCC} "${BUILD_DIR}/${SRC}.o" "${HELPERS[@]}" -lm -lrt -o "${BIN}"
echo "  Generated: ${BIN}"
echo "  agg buffers: $(grep -c 'shmem_agg_buf' "${BUILD_DIR}/${SRC}.ll" 2>/dev/null || echo 0)   bulk putmem/getmem: $(grep -cE 'shmem_(put|get)mem' "${BUILD_DIR}/${SRC}.ll" 2>/dev/null || echo 0)"

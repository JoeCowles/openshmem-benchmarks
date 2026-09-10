#!/bin/bash
# build_ssca1.sh — Compile SSCA1 OpenSHMEM benchmark through the
# ClangIR → OpenSHMEM MLIR → LLVM IR pipeline.
#
# Usage:
#   ./build_ssca1.sh              # build bin/ssca1
#   ./build_ssca1.sh --run        # build and run with scale=16, 2 PEs
#   ./build_ssca1.sh --run -n 4 --scale 18
#
# Env:
#   USE_FUSE_ATOMICS=1|0             # enable/disable --openshmem-fuse-atomics
#   USE_MESSAGE_AGGREGATION=1|0       # enable/disable --openshmem-message-aggregation
#   USE_REMOTE_UPDATE_COMBINING=1|0   # enable/disable --openshmem-remote-update-combining
#   USE_SYNC_MINIMIZATION=1|0         # enable/disable --openshmem-sync-minimization
#
# Output: bin/ssca1

set -e

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
CLANG=/mnt/DISCL/home/jcowles/MLIR_testing/clangir/build-new/bin/clang
CIR_OPT=/mnt/DISCL/home/jcowles/MLIR_testing/clangir/build-new/bin/cir-opt
SHMEM_OPT=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/build-incubator/bin/shmem-mlir-opt
MLIR_OPT=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/clangir/build-main/bin/mlir-opt
MLIR_TRANSLATE=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/clangir/build-main/bin/mlir-translate
LLC=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/clangir/build-main/bin/llc

SOS_DIR=/mnt/DISCL/home/jcowles/MLIR_testing/openshmem-mlir/openshmem-runtime/SOS-v1.5.2
OSHCC=${SOS_DIR}/bin/oshcc
SHMEM_INC=${SOS_DIR}/include

ATOMIC_LIB_DIR=/opt/apps/nfs/spack-v0.23/opt/spack/linux-rocky9-zen4/gcc-11.4.1/gcc-runtime-11.4.1-7hex6dyh2ttbdeywfkq5vbsinmnhjoub/lib

# ---------------------------------------------------------------------------
# SSCA1 source layout
# ---------------------------------------------------------------------------
SSCA1_DIR=/mnt/DISCL/home/jcowles/MLIR_testing/SSCA1_MLIR

SSCA1_SOURCES=(
  "${SSCA1_DIR}/parameters.c"
  "${SSCA1_DIR}/gen_sim_matrix.c"
  "${SSCA1_DIR}/gen_scal_data.c"
  "${SSCA1_DIR}/pairwise_align.c"
  "${SSCA1_DIR}/glibc_sort.c"
  "${SSCA1_DIR}/scan_backwards.c"
  "${SSCA1_DIR}/util.c"
  "${SSCA1_DIR}/main.c"
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
DO_RUN=0
RUN_NP=2
RUN_SCALE=16
USE_FUSE_ATOMICS="${USE_FUSE_ATOMICS:-1}"
USE_MESSAGE_AGGREGATION="${USE_MESSAGE_AGGREGATION:-0}"
USE_REMOTE_UPDATE_COMBINING="${USE_REMOTE_UPDATE_COMBINING:-0}"
USE_SYNC_MINIMIZATION="${USE_SYNC_MINIMIZATION:-0}"
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)    DO_RUN=1; shift ;;
    -n)       RUN_NP="$2";    shift 2 ;;
    --scale)  RUN_SCALE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

FUSE_ATOMIC_ARGS=()
if [[ "${USE_FUSE_ATOMICS}" == "1" ]]; then
  FUSE_ATOMIC_ARGS+=(--openshmem-fuse-atomics)
fi

MESSAGE_AGGREGATION_ARGS=()
if [[ "${USE_MESSAGE_AGGREGATION}" == "1" ]]; then
  MESSAGE_AGGREGATION_ARGS+=(--openshmem-message-aggregation)
fi

REMOTE_UPDATE_COMBINING_ARGS=()
if [[ "${USE_REMOTE_UPDATE_COMBINING}" == "1" ]]; then
  REMOTE_UPDATE_COMBINING_ARGS+=(--openshmem-remote-update-combining)
fi

SYNC_MINIMIZATION_ARGS=()
if [[ "${USE_SYNC_MINIMIZATION}" == "1" ]]; then
  SYNC_MINIMIZATION_ARGS+=(--openshmem-sync-minimization)
fi

local_length=$(awk -v scale="${RUN_SCALE}" -v np="${RUN_NP}" 'BEGIN {
  len = 2.0 ^ (scale / 2.0);
  if (len == int(len)) {
    ceil_len = len;
  } else {
    ceil_len = int(len) + 1;
  }
  size = ceil_len + 80;
  local_len = int(size / np);
  print local_len;
}')
# ---------------------------------------------------------------------------
# Build directories
# ---------------------------------------------------------------------------
BUILD_DIR="${BUILD_DIR:-/mnt/DISCL/home/jcowles/MLIR_testing/SSCA1_MLIR/build-mlir}"
mkdir -p \
  "${BUILD_DIR}/cir" \
  "${BUILD_DIR}/flat_cir" \
  "${BUILD_DIR}/openshmem_mlir" \
  "${BUILD_DIR}/llvm_mlir" \
  "${BUILD_DIR}/llvm_ir" \
  "${BUILD_DIR}/obj" \
  "${BUILD_DIR}/bin"

BIN="${BUILD_DIR}/bin/ssca1"

# ---------------------------------------------------------------------------
# Common compiler flags (matches parameters and headers)
# ---------------------------------------------------------------------------
# OPT flags setting
OPT_FLAGS=("-O3")
if [[ "${USE_OPT_FLAGS:-1}" == "0" ]]; then
  OPT_FLAGS=("-O0")
fi

CFLAGS=(
  -std=c99
  -g
  "${OPT_FLAGS[@]}"
  -Wno-error
  -Wno-incompatible-pointer-types
  -Wall
  -DUSE_SHMEM
  -I"${SSCA1_DIR}"
  -I"${SHMEM_INC}"
)

# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------
OBJ_FILES=()

compile_file() {
  local src="$1"
  local base
  base="$(basename "${src}" .c)"

  echo "  [${base}]"

  # Step 1: C → CIR
  ${CLANG} "${CFLAGS[@]}" -fclangir -emit-cir "${src}" \
    -o "${BUILD_DIR}/cir/${base}.cir" 2>&1 || return 1

  # Step 2: CIR → flattened CIR
  ${CIR_OPT} \
    --cir-goto-solver \
    --cir-flatten-cfg \
    --cir-abi-lowering \
    "${BUILD_DIR}/cir/${base}.cir" \
    -o "${BUILD_DIR}/flat_cir/${base}.flat.cir" 2>&1 || return 1

  # Steps 3-5 combined: CIR → OpenSHMEM → Lower to LLVM dialect
  ${SHMEM_OPT} \
    --convert-cir-to-openshmem \
    "${FUSE_ATOMIC_ARGS[@]}" \
    "${REMOTE_UPDATE_COMBINING_ARGS[@]}" \
    "${MESSAGE_AGGREGATION_ARGS[@]}" \
    "${SYNC_MINIMIZATION_ARGS[@]}" \
    --convert-openshmem-to-llvm \
    --allow-unregistered-dialect \
    --cir-to-llvm \
    --reconcile-unrealized-casts \
    --mlir-print-op-generic \
    "${BUILD_DIR}/flat_cir/${base}.flat.cir" \
    -o "${BUILD_DIR}/llvm_mlir/${base}.llvm.mlir" 2>&1 || return 1

  # Step 6: LLVM MLIR → LLVM IR
  ${MLIR_TRANSLATE} \
    --allow-unregistered-dialect \
    --mlir-to-llvmir \
    "${BUILD_DIR}/llvm_mlir/${base}.llvm.mlir" \
    -o "${BUILD_DIR}/llvm_ir/${base}.ll" 2>&1 || return 1

  # Step 7: LLVM IR → Object
  ${LLC} \
    -O3 \
    -filetype=obj \
    -relocation-model=pic \
    "${BUILD_DIR}/llvm_ir/${base}.ll" \
    -o "${BUILD_DIR}/obj/${base}.o" 2>&1 || return 1

  OBJ_FILES+=("${BUILD_DIR}/obj/${base}.o")
}

# ---------------------------------------------------------------------------
# Compile all sources
# ---------------------------------------------------------------------------
echo "=== SSCA1 OpenSHMEM MLIR Build ==="
echo "Sources: ${#SSCA1_SOURCES[@]} files"
echo ""
echo "Compiling..."

FAIL=0
for src in "${SSCA1_SOURCES[@]}"; do
  compile_file "${src}" || { echo "  FAILED: ${src}"; FAIL=1; }
done

if [[ ${FAIL} -eq 1 ]]; then
  echo ""
  echo "ERROR: One or more files failed to compile." >&2
  exit 1
fi

echo ""

# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------
echo "Linking ${BIN}..."
SHMEM_CC="${CLANG}" \
${OSHCC} \
  "${OBJ_FILES[@]}" \
  -Wl,-rpath,"${ATOMIC_LIB_DIR}" \
  -lm -lrt \
  -o "${BIN}"

echo "  Generated: ${BIN}"
echo ""
echo "=== Build complete ==="
echo ""

# ---------------------------------------------------------------------------
# Optional run
# ---------------------------------------------------------------------------
if [[ ${DO_RUN} -eq 1 ]]; then
  echo "Running: SCALE=${RUN_SCALE}, ${RUN_NP} PEs..."
  SCALE="${RUN_SCALE}" \
  PATH="${SOS_DIR}/bin:${PATH}" \
  LD_LIBRARY_PATH="${SOS_DIR}/lib:${LD_LIBRARY_PATH:-}" \
  OSHRUN_LAUNCHER="srun --mpi=pmi2" \
  SHMEM_SYMMETRIC_HEAP_SIZE=12000000000 \
  SHMEM_OFI_PROVIDER=sockets \
  oshrun -n "${RUN_NP}" setarch $(uname -m) -R "${BIN}"
fi

# OpenSHMEM Benchmarks

A collection of OpenSHMEM benchmarks and mini-apps used to evaluate compiler-driven
communication optimizations (message aggregation, inspector–executor gather, atomic
fusion, synchronization minimization) implemented as MLIR passes.

Each benchmark is compiled through the same pipeline:

```
C source --clang--> ClangIR --cir-opt--> flat CIR --shmem-mlir-opt--> OpenSHMEM MLIR
        --> LLVM MLIR --mlir-translate--> LLVM IR --llc--> object --oshcc--> binary
```

Every benchmark can be built twice from *identical source*: once with the OpenSHMEM
passes disabled (the **control** / baseline) and once with them enabled. Several also
carry a hand-written "oracle" variant, so compiler output can be compared against what
a human would have written by hand.

## Benchmarks

| Directory | What it is | Pass exercised |
|---|---|---|
| `ssca1/` | SSCA #1 — Smith-Waterman pairwise sequence alignment | scalar-to-bulk aggregation, atomic fusion, sync minimization |
| `graph500/` | Graph500 BFS (SHMEM), as a submodule | inspector–executor / aggregation |
| `stencil/` | Integer Jacobi stencil, regular and "dense" halo variants | scalar-to-bulk aggregation |
| `prk/` | Parallel Research Kernels — `prk_stencil`, `prk_p2p` (SHMEM) | scalar-to-bulk, block aggregation |
| `indexgather/` | Irregular gather `A[B[i]]` microbenchmark | inspector–executor (Bulk / Pack) |
| `bale-ig/` | bale index-gather kernel, extracted as a standalone driver | inspector–executor (Pack) |
| `bale_src/` | Upstream bale source tree (vendored, see its own `LICENSE`) | reference / source of `bale-ig` |
| `bspmm/` | NWChem BSPMM mini-app (block sparse matmul, get–compute–update) | aggregation (already-bulk baseline) |
| `tmlqcd/` | tmLQCD halo exchange (`xchange_field`), extracted | block aggregation |

## Building

Each benchmark has a `build_*.sh` that drives the whole pipeline. The pass toggles are
environment variables, so control and optimized builds come from one script:

```sh
# stencil: baseline vs. compiler aggregation
BUILD_DIR=build-reg-ctrl USE_SCALAR_TO_BULK=0 ./stencil/build_stencil.sh
BUILD_DIR=build-reg-agg  USE_SCALAR_TO_BULK=1 ./stencil/build_stencil.sh

# indexgather: naive / hand-aggregated oracle / compiler inspector-executor
BUILD_DIR=build-ctrl                          ./indexgather/build_indexgather.sh
BUILD_DIR=build-man USE_MANUAL_AGG=1          ./indexgather/build_indexgather.sh
BUILD_DIR=build-ie  USE_INSPECTOR_EXECUTOR=1  ./indexgather/build_indexgather.sh

# PRK: same source, passes off vs. on
SRC=prk_stencil PASS=off ./prk/build_prk.sh
SRC=prk_stencil PASS=stb ./prk/build_prk.sh
```

Each script's header comment documents its full set of environment variables
(problem-size `-D` overrides, pass flags, output directory).

### Toolchain

The build scripts currently reference the toolchain by **absolute path** at the top of
each file — a ClangIR build, an `shmem-mlir-opt` build, and Sandia OpenSHMEM (SOS)
v1.5.2. Edit those variables at the top of each `build_*.sh` to point at your own
builds:

```sh
CLANG=.../clangir/build/bin/clang
CIR_OPT=.../clangir/build/bin/cir-opt
SHMEM_OPT=.../openshmem-mlir/build/bin/shmem-mlir-opt
MLIR_TRANSLATE=.../build/bin/mlir-translate
LLC=.../build/bin/llc
SOS_DIR=.../SOS-v1.5.2
```

Generated output (`build*/`, `val-*/`, IR, objects, binaries) is gitignored — everything
in it is reproducible from the scripts.

## Cloning

`graph500/` is a submodule:

```sh
git clone --recurse-submodules <this repo>
# or, in an existing clone:
git submodule update --init --recursive
```

## Licensing

The vendored upstream trees keep their original licenses — see `bale_src/LICENSE`,
`bspmm/README.md`, and the `COPYING` / `LICENSE` files inside the graph500 submodule.

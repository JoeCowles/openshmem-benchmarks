/*
   This file is part of SSCA1.

   Copyright (C) 2008-2015, UT-Battelle, LLC.

   This product includes software produced by UT-Battelle, LLC under Contract No.
   DE-AC05-00OR22725 with the Department of Energy.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the New BSD 3-clause software license (LICENSE).

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   LICENSE for more details.

   For more information please contact the SSCA1 developers at:
   bakermb@ornl.gov
*/

#ifndef __UTIL_H
#define __UTIL_H

#include "pairwise_align.h"
#include "types.h"
#include <stdio.h>

#ifdef USE_MPI3
#include <mpi.h>
extern MPI_Comm world;
extern MPI_Win window;
extern void *window_base;
extern size_t window_size;
extern void *next_window_address;
extern MPI_Request request;
#else
#ifdef SGI_SHMEM
#include <mpp/shmem.h>
#else
#include <shmem.h>
#endif
#endif

#ifdef USE_MPI3
#define SHORT_GET(target, source, num_elems, rank)	MPI_Get(target, num_elems, MPI_SHORT, rank, (void *)source - window_base, num_elems, MPI_SHORT, window); QUIET()
#else
#define SHORT_GET(target, source, num_elems, pe)	shmem_short_get(target, source, num_elems, pe)
#endif

#ifdef USE_MPI3
#define SHORT_GET_NB(target, source, num_elems, rank)	MPI_Get(target, num_elems, MPI_SHORT, rank, (void *)source - window_base, num_elems, MPI_SHORT, window)
#else
#define SHORT_GET_NB(target, source, num_elems, pe)	shmem_short_get_nbi(target, source, num_elems, pe)
#endif

#ifdef USE_MPI3
#define LONG_GET(target, source, num_elems, rank)	MPI_Get(target, num_elems, MPI_LONG, rank, (void *)source - window_base, num_elems, MPI_LONG, window); QUIET()
#else
#define LONG_GET(target, source, num_elems, pe)		shmem_long_get((long*)target, (long*)source, num_elems, pe)
#endif

#ifdef USE_MPI3
#define GETMEM(target, source, length, rank)		MPI_Get(target, length, MPI_BYTE, rank, (void *)source - window_base, length, MPI_BYTE, window); QUIET()
#else
#define GETMEM(target, source, length, pe)		shmem_getmem(target, source, length, pe)
#endif

#ifdef USE_MPI3
#define SHORT_PUT(target, source, num_elems, rank)	MPI_Put(source, num_elems, MPI_SHORT, rank, (void *)target - window_base, num_elems, MPI_SHORT, window); QUIET()
#else
#define SHORT_PUT(target, source, num_elems, pe)	shmem_short_put(target, source, num_elems, pe)
#endif

#ifdef USE_MPI3
#define QUIET()		MPI_Win_flush_all(window)
#else
#define QUIET()		shmem_quiet()
#endif

#ifdef USE_MPI3
#define BARRIER_ALL()	QUIET(); MPI_Barrier(MPI_COMM_WORLD)
#else
#define BARRIER_ALL()	shmem_barrier_all()
#endif

/* Communication/computation overlap for the prefetch stream (USE_CTX_OVERLAP).
 *
 * A dedicated OpenSHMEM context carries the input-prefetch stream so it can be
 * completed (CTX_QUIET = shmem_ctx_quiet) independently of the output-scatter
 * stream on the default context, instead of a global shmem_quiet draining
 * everything. The ClangIR->OpenSHMEM-MLIR pipeline lowers these via the typed
 * context-RMA patterns in cir/lib/Rewriters/RewriteRMA.cpp (shmem_ctx_short_get_nbi
 * -> openshmem.ctx_get_nbi -> shmem_ctx_get16_nbi). The MPI-3 fallbacks ignore
 * the context (single window). */
#ifdef USE_MPI3
#define CTX_CREATE(ctxptr)	(*(ctxptr) = 0, 0)
#define CTX_DESTROY(ctx)	((void)0)
#define CTX_SHORT_GET_NB(ctx, target, source, num_elems, rank) \
	MPI_Get(target, num_elems, MPI_SHORT, rank, (void *)source - window_base, num_elems, MPI_SHORT, window)
#define CTX_QUIET(ctx)		MPI_Win_flush_all(window)
#else
/* Best-performing overlap on this SOS+verbs runtime: plain non-blocking prefetch
 * on the default context + shmem_quiet (+5..38% over aggregation). The typed
 * context-RMA lowering IS implemented (cir/lib/Rewriters/RewriteRMA.cpp) and
 * verified to run, but empirically the SOS context-RMA path (shmem_ctx_get16_nbi
 * + shmem_ctx_quiet) is slower here — a separate context is a distinct OFI
 * endpoint not progressed during compute (-30..44%), and even the ctx API on the
 * default context regresses at high PE (-7..11%). To switch to the literal
 * context calls, use shmem_ctx_short_get_nbi/shmem_ctx_quiet here. */
#define CTX_CREATE(ctxptr)	(*(ctxptr) = SHMEM_CTX_DEFAULT, 0)
#define CTX_DESTROY(ctx)	((void)(ctx))
#define CTX_SHORT_GET_NB(ctx, target, source, num_elems, pe) \
	((void)(ctx), shmem_short_get_nbi((target), (source), (num_elems), (pe)))
#define CTX_QUIET(ctx)		((void)(ctx), shmem_quiet())
#endif

#ifdef USE_MPI3
static inline int malloc_all(size_t size, void **address) {
  *address = next_window_address;
  next_window_address += size;
  MPI_Barrier(MPI_COMM_WORLD);
  if (next_window_address - window_base > window_size) {
    printf("ran out of memory!\n");
    return -1;
  } else
    return 0;
}
#else
static inline int malloc_all(size_t size, void **address) {
  *address = shmalloc(size);
  if (*address == NULL)
    return -1;
  else
    return 0;
}
#endif

#ifdef USE_MPI3
#define FREE_ALL(address) /* unable to free memory like this */
#else
#define FREE_ALL(address) shfree(address)
#endif

static inline int global_index_to_rank(const seq_t *in, const index_t codon_index){
  return codon_index / in->local_size;
}

static inline int global_index_to_local_index(const seq_t *in, const index_t codon_index){
  return codon_index % in->local_size;
}

static inline void fetch_from_seq(const seq_t *in, index_t const codon_index, codon_t *out){
  int target_ep = global_index_to_rank(in,codon_index);
  int local_index = global_index_to_local_index(in,codon_index);
  short *typed_seq = (short *)in->sequence;
  SHORT_GET((short *)out, &(typed_seq[local_index]), 1, target_ep);
}

static inline void fetch_from_seq_nb(const seq_t *in, index_t const codon_index, codon_t *out){
  int target_ep = global_index_to_rank(in,codon_index);
  int local_index = global_index_to_local_index(in,codon_index);
  short *typed_seq = (short *)in->sequence;
  SHORT_GET_NB((short *)out, &(typed_seq[local_index]), 1, target_ep);
}

static inline void write_to_seq(const seq_t *in, const index_t codon_index, codon_t data){
  int target_ep = global_index_to_rank(in,codon_index);
  int local_index = global_index_to_local_index(in,codon_index);
  short *typed_seq = (short *)in->sequence;
  short typed_data = (short)data;
  SHORT_PUT(&(typed_seq[local_index]), &typed_data, 1, target_ep);
}

#ifdef USE_MPI3
#define WAIT_NB() QUIET()
#else
#define WAIT_NB() QUIET()
#endif

void distribute_rng_seed(unsigned int new_seed);
void seed_rng(int adjustment);
void touch_memory(void *mem, index_t size);
index_t scrub_hyphens(good_match_t *A, seq_t *dest, seq_t *source, index_t length);
void assemble_acid_chain(good_match_t *A, char *result, seq_t *chain, index_t length);
void assemble_codon_chain(good_match_t *A, char *result, seq_t *chain, index_t length);
score_t simple_score(good_match_t *A, seq_t *main, seq_t *match);
seq_t *alloc_global_seq(index_t seq_size);
seq_t *alloc_local_seq(index_t seq_size);
void free_global_seq(seq_t *doomed);
void free_local_seq(seq_t *doomed);

#endif

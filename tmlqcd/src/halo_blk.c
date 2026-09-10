/* halo_blk.c — naive block halo exchange, the message-aggregation target.
 *
 * Models a stencil boundary exchange written the naive way: instead of one bulk
 * transfer per neighbor, it puts the boundary one fixed-size BLOCK at a time in
 * a counted loop (like tmLQCD's xchange_field.c, which puts one spinor / one
 * lattice row per iteration). Two source flavors, selected at compile time:
 *
 *   default        : CONTIGUOUS source  src + i*W   (blocks collapse to one put)
 *   -DGATHERED     : GATHERED   source  src + idx[i]  (tmLQCD's g_field_z_ipt[k])
 *
 * The message-aggregation pass should turn the whole loop into ONE bulk put of
 * NBLK*W elements to the (loop-invariant) neighbor PE.
 *
 *   -DNBLK=<n>  number of boundary blocks (loop trip count)   default 8192
 *   -DW=<n>     doubles per block (block size)                default 24 (a spinor)
 *   -DREPS=<n>  repeat the exchange REPS times (timing)       default 2000
 */
#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
static double now(void){struct timeval tv;gettimeofday(&tv,0);return tv.tv_sec+tv.tv_usec*1e-6;}

#ifndef NBLK
#define NBLK 8192
#endif
#ifndef W
#define W 24
#endif
#ifndef REPS
#define REPS 2000
#endif

int main(void) {
  start_pes(0);
  int me = shmem_my_pe();
  int npes = shmem_n_pes();
  int nbr = (me + 1) % npes;            /* one loop-invariant neighbor */

  long n = (long)NBLK * W;
  long *src = (long *)shmem_malloc(sizeof(long) * n);
  long *dst = (long *)shmem_malloc(sizeof(long) * n);
  for (long i = 0; i < n; i++) src[i] = (long)(me + 1) * 1000 + i;

#ifdef GATHERED
  int *idx = (int *)malloc(sizeof(int) * NBLK);
  for (int i = 0; i < NBLK; i++) idx[i] = ((i * 2654435761u) % NBLK) * W; /* scattered */
#endif

  int nb = NBLK;            /* runtime trip count (analyzeLoop needs a variable bound) */
  shmem_barrier_all();
  double t0 = now();
  /* warm + timed reps of the naive block halo */
  for (int r = 0; r < REPS; r++) {
    for (int i = 0; i < nb; i++) {
#ifdef GATHERED
      shmem_long_put(dst + i * W, src + idx[i], W, nbr);
#else
      shmem_long_put(dst + i * W, src + i * W, W, nbr);
#endif
    }
    shmem_barrier_all();
  }

  double telapsed = now() - t0;
  long acc = 0.0;
  for (long i = 0; i < n; i++) acc += dst[i] * (long)(1 + (i % 131)); /* position-sensitive */
  if (me == 0) printf("halo_blk done: NBLK=%d W=%d REPS=%d npes=%d acc=%ld time=%.4fs\n", NBLK, W, REPS, npes, acc, telapsed);
  shmem_free(src);
  shmem_free(dst);
  return 0;
}

/* stencil.c — distributed 2-D 5-point integer Jacobi stencil, 1-D row-block
 * decomposition over OpenSHMEM PEs.  Second benchmark for the openshmem-mlir
 * message-aggregation pass.
 *
 * The grid is NROWS x NCOLS ints; NROWS is split into P contiguous row blocks,
 * one per PE (LROWS = NROWS / P rows each), stored in the symmetric array
 * `grid`.  A cell update needs its 4 neighbors; the only NON-local neighbors are
 * the row just above a PE's top edge (owned by PE me-1) and the row just below
 * its bottom edge (owned by PE me+1).
 *
 * The NAIVE halo exchange (this file, default) reads that neighbor row ONE
 * ELEMENT AT A TIME in a column loop:
 *
 *     for (c = 0; c < NCOLS; c++) {
 *       short v;
 *       shmem_short_get(&v, &grid[(LROWS-1)*NCOLS + c], 1, up);  // last row of me-1
 *       top_halo[c] = v;
 *     }
 *
 * i.e. NCOLS tiny 2-byte remote gets per halo per step — latency-bound on RDMA.
 * The remote index is affine in the loop IV c (invariant offset (LROWS-1)*NCOLS,
 * stride +1) and the target PE (up = me-1) is loop-INVARIANT, so this is exactly
 * the --openshmem-scalar-to-bulk INVARIANT-PE pattern: the pass coalesces the
 * whole column loop into ONE bulk `shmem_short_get` of NCOLS elements before the
 * loop.  Because the halo loop only READS the remote array (it writes the local
 * top_halo/bot_halo), the pass fires WITHOUT assume-independent (unlike SSCA1,
 * whose wavefront both reads and writes the remote matrices).
 *
 * Correctness is validated by a global checksum, compared bit-for-bit between the
 * control (naive) and compiler-aggregated builds.
 */
#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef NCOLS
#define NCOLS 4096
#endif
#ifndef LROWS      /* local rows per PE */
#define LROWS 64
#endif
#ifndef NSTEPS
#define NSTEPS 50
#endif

/* Symmetric objects (global/static are symmetric in OpenSHMEM). */
static short grid[LROWS * NCOLS];
static short next[LROWS * NCOLS];
static short top_halo[NCOLS];
static short bot_halo[NCOLS];
static long  partial_sum; /* per-PE checksum, read by PE 0 */

static inline int idx(int r, int c) { return r * NCOLS + c; }

int main(void) {
  start_pes(0); /* SSCA1's proven init path; shmem_init+shmem_finalize triggers a
                   buggy openshmem.region wrap in convert-cir-to-openshmem */
  int me = shmem_my_pe();
  int np = shmem_n_pes();

  /* Deterministic initial field. */
  for (int r = 0; r < LROWS; r++)
    for (int c = 0; c < NCOLS; c++)
      grid[idx(r, c)] = (short)(((me * LROWS + r) * 131 + c * 17) & 0xff);

  shmem_barrier_all();
  struct timeval t0, t1;
  gettimeofday(&t0, 0);

  const int up = me - 1;   /* loop-invariant target PEs */
  const int down = me + 1;
  /* Runtime loop bound: the aggregation pass requires the loop's upper bound to be
     a loaded variable (a counted loop `i < n`), not a compile-time constant. */
  long ncols = NCOLS; /* 64-bit (like SSCA1's index_t=uint64_t) so the pass's
                         trip-count lowers cleanly to `index` (i64). */
  /* Access the symmetric grid through a pointer so the remote element address is
     pointer arithmetic (cir.ptr_stride, what the pass matches) rather than direct
     global-array indexing (cir.get_element).  g points into the symmetric object,
     so &g[k] is a valid symmetric address on any PE. */
  short *g = grid;

  for (int step = 0; step < NSTEPS; step++) {
    /* ---- Halo exchange: naive per-column remote gets (pass aggregates these) ---- */
    if (me > 0) {
      for (long c = 0; c < ncols; c++) {
        short v;
        shmem_short_get(&v, &g[(long)(LROWS - 1) * NCOLS + c], 1, up); /* last row of PE me-1 */
        top_halo[c] = v;
      }
    } else {
      for (int c = 0; c < NCOLS; c++)
        top_halo[c] = grid[idx(0, c)]; /* reflective top boundary */
    }
    if (me < np - 1) {
      for (long c = 0; c < ncols; c++) {
        short v;
        shmem_short_get(&v, &g[c], 1, down); /* first row of PE me+1 (idx(0,c)=c) */
        bot_halo[c] = v;
      }
    } else {
      for (int c = 0; c < NCOLS; c++)
        bot_halo[c] = grid[idx(LROWS - 1, c)]; /* reflective bottom boundary */
    }
    shmem_barrier_all();

    /* ---- 5-point integer stencil (purely local, uses the gathered halos) ---- */
    for (int r = 0; r < LROWS; r++) {
      for (int c = 0; c < NCOLS; c++) {
        int u = (r == 0)         ? top_halo[c] : grid[idx(r - 1, c)];
        int d = (r == LROWS - 1) ? bot_halo[c] : grid[idx(r + 1, c)];
        int l = (c == 0)         ? grid[idx(r, c)] : grid[idx(r, c - 1)];
        int rt = (c == NCOLS - 1) ? grid[idx(r, c)] : grid[idx(r, c + 1)];
        int ctr = grid[idx(r, c)];
        next[idx(r, c)] = (short)((u + d + l + rt + ctr) / 5);
      }
    }
    for (int i = 0; i < LROWS * NCOLS; i++)
      grid[i] = next[i];
    shmem_barrier_all();
  }

  gettimeofday(&t1, 0);
  double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1e6;

  /* ---- Global checksum (manual gather; PE-indexed get is left alone by pass) ---- */
  long s = 0;
  for (int i = 0; i < LROWS * NCOLS; i++)
    s += grid[i];
  partial_sum = s;
  shmem_barrier_all();
  if (me == 0) {
    long total = 0;
    for (int p = 0; p < np; p++) {
      long x;
      shmem_long_get(&x, &partial_sum, 1, p);
      total += x;
    }
    printf("Begining Kernel 1 execution.\n"); /* marker for the k1max parser */
    printf("Stencil done. steps=%d local=%dx%d np=%d checksum=%ld\n",
           NSTEPS, LROWS, NCOLS, np, total);
    printf("\tElapsed time: 0 hour(s), 0 minute(s), %d second(s), %d milliseconds, %d micro second(s).\n",
           (int)sec, (int)((sec - (int)sec) * 1000),
           (int)(((sec * 1000) - (long)(sec * 1000)) * 1000));
  }
  return 0;
}

/* indexgather.c — distributed irregular GATHER (bale `indexgather` / Arkouda
 * gather pattern) for the openshmem-mlir INSPECTOR-EXECUTOR pass.
 *
 * A read-only table of GLOBAL size (np * LOCAL_LEN) longs is block-distributed:
 * PE p owns `table[0..LOCAL_LEN)`, holding the values of global indices
 * [p*LOCAL_LEN, (p+1)*LOCAL_LEN).  Each PE issues NREQ requests; request i wants
 * the table value at a data-dependent global index `index[i]`, whose owner PE is
 * `index[i] / LOCAL_LEN` — a DIFFERENT, data-dependent PE every iteration.
 *
 *   NAIVE (this file, default):  one 8-byte remote get per request:
 *       for (i=0;i<n;i++){ g=index[i]; tgt[i]=get(&table[g%L], g/L); }
 *   -> NREQ tiny latency-bound gets.  This is the exact shape scalar-to-bulk
 *      CORRECTLY DECLINES (data-dependent PE, non-affine index) and the
 *      inspector-executor pass targets.
 *
 *   MANUAL oracle (-DUSE_MANUAL_AGG):  Titanium "Bulk" fetch method — one bulk
 *      get of each owner PE's whole block into a local mirror, then index the
 *      mirror locally.  np bulk gets replace NREQ tiny gets.  This is the
 *      hand-aggregated ground truth the pass must match, bit-for-bit.
 *
 * Correctness: a position-sensitive checksum over tgt[], reduced to PE 0, must
 * be identical across control / manual / inspector-executor builds.
 */
#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef LOCAL_LEN          /* longs owned per PE (block size) */
#define LOCAL_LEN (1 << 16)
#endif
#ifndef NREQ               /* gather requests per PE */
#define NREQ (1 << 20)
#endif
#ifndef REPS               /* repeat the gather to amortize setup / amplify signal */
#define REPS 20
#endif

/* Symmetric: the distributed table (statics are symmetric in OpenSHMEM). */
static long table[LOCAL_LEN];
static long partial_sum;   /* per-PE checksum, read by PE 0 */

/* Deterministic table value for global index g (so results are reproducible and
   independent of decomposition). */
static inline long tabval(long g) { return (g * 1000003L + 12345L) & 0x00ffffffL; }

/* ---- The naive gather kernel: exactly the phase7 inspector-executor shape ----
 * Runtime loop bound `n`, pointer addressing (cir.ptr_stride), 64-bit index —
 * the preconditions the pass matches.  Kept in its own function so the pass has
 * a clean loop to transform. */
#ifndef USE_MANUAL_AGG
static void gather_kernel(long *restrict tgt, long *restrict table_p,
                          const long *restrict index, long n, long local_len) {
  for (long i = 0; i < n; i++) {
    long g = index[i];            /* data-dependent global index */
    long pe = g / local_len;      /* => different owner PE every iteration */
    long loc = g % local_len;
    long v;
    shmem_long_get(&v, &table_p[loc], 1, pe);
    tgt[i] = v;
  }
}
#else
/* Hand-aggregated oracle (Titanium "Bulk" method): replicate each owner block
   into a local mirror with one bulk get per PE, then index locally. */
static void gather_kernel(long *restrict tgt, long *restrict table_p,
                          const long *restrict index, long n, long local_len) {
  int np = shmem_n_pes();
  long *mirror = (long *)malloc((size_t)np * local_len * sizeof(long));
  for (int p = 0; p < np; p++)
    shmem_long_get(&mirror[(long)p * local_len], table_p, local_len, p);
  for (long i = 0; i < n; i++)
    tgt[i] = mirror[index[i]];
  free(mirror);
}
#endif

int main(void) {
  start_pes(0);
  int me = shmem_my_pe();
  int np = shmem_n_pes();
  long local_len = LOCAL_LEN;
  long gsize = (long)np * local_len;   /* global table size */
  long nreq = NREQ;

  /* Fill this PE's block of the distributed table. */
  for (long j = 0; j < local_len; j++)
    table[j] = tabval((long)me * local_len + j);

  /* Deterministic per-PE request stream (LCG); indices span the whole table. */
  long *index = (long *)malloc((size_t)nreq * sizeof(long));
  long *tgt = (long *)malloc((size_t)nreq * sizeof(long));
  unsigned long st = 0x9e3779b97f4a7c15UL ^ ((unsigned long)me * 0xff51afd7ed558ccdUL + 1);
  for (long i = 0; i < nreq; i++) {
    st = st * 6364136223846793005UL + 1442695040888963407UL;
    index[i] = (long)((st >> 17) % (unsigned long)gsize);
  }

  shmem_barrier_all();
  struct timeval t0, t1;
  gettimeofday(&t0, 0);

  long *tp = table;   /* pointer into the symmetric object (valid on any PE) */
  for (int rep = 0; rep < REPS; rep++)
    gather_kernel(tgt, tp, index, nreq, local_len);

  gettimeofday(&t1, 0);
  double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1e6;

  /* Position-sensitive local checksum (catches order/placement bugs). */
  long s = 0;
  for (long i = 0; i < nreq; i++)
    s += tgt[i] * (1 + (i % 131));
  partial_sum = s;
  shmem_barrier_all();

  if (me == 0) {
    long total = 0;
    for (int p = 0; p < np; p++) {
      long x;
      shmem_long_get(&x, &partial_sum, 1, p);
      total += x;
    }
    printf("Begining Kernel 1 execution.\n"); /* k1max parser marker */
    printf("IndexGather done. np=%d local_len=%ld nreq=%ld reps=%d checksum=%ld\n",
           np, local_len, nreq, REPS, total);
    printf("\tElapsed time: 0 hour(s), 0 minute(s), %d second(s), %d milliseconds, %d micro second(s).\n",
           (int)sec, (int)((sec - (int)sec) * 1000),
           (int)(((sec * 1000) - (long)(sec * 1000)) * 1000));
  }
  free(index);
  free(tgt);
  return 0;
}

/* bale_ig.c — test vehicle for the inspector-executor PACK pass on BALE's real
 * indexgather (ig) kernel.
 *
 * The optimization target — the ig_agp kernel and the libgetput SHMEM get
 * wrapper — is BALE's VERBATIM source (github.com/jdevinney/bale, Institute for
 * Defense Analyses, BSD). Only the surrounding driver (symmetric table setup,
 * random request generation, correctness check) is harness scaffolding, and it
 * follows bale's own ig.upc setup and ig_check_and_zero criterion exactly:
 *   table (cyclic-distributed): ltable[i] = -(i*THREADS + MYTHREAD + 1)
 *   requests:                   index[i] = uniform random in [0, tab_siz)
 *   correctness:                tgt[i] == -(index[i] + 1)   (== table[index[i]])
 *
 * bale's naive model is single-word gets to random remote PEs (pe = idx % THREADS,
 * cyclic) — exactly the irregular A[B[i]] pattern the inspector-executor pass
 * targets and scalar-to-bulk declines.
 */
#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

/* ===== BALE libgetput SHMEM backend (VERBATIM from libgetput.upc / .h) ===== */
#define shmem_int64_g(addr, pe) shmem_longlong_g((const long long *)(addr), (pe))
/* libgetput.upc:270 — verbatim (static so the pass's inline-comm-helpers exposes
   the get + cyclic index math inside ig_agp's loop). */
static int64_t lgp_shmem_read_upc_array_int64(const int64_t *addr, size_t index,
                                              size_t blocksize) {
  int pe;
  size_t local_index;
  int64_t *local_ptr;

  pe = index % shmem_n_pes();
  local_index = (index / shmem_n_pes()) * blocksize;

  local_ptr = (int64_t *)(((char *)addr) + local_index);

  return shmem_int64_g(local_ptr, pe);
}
#define lgp_get_int64(array, index) \
  (lgp_shmem_read_upc_array_int64((array), (index), sizeof(int64_t)))

/* ===== BALE ig_agp kernel (VERBATIM hot loop from apps/ig_src/ig_agp.upc) =====
   bale-infrastructure calls (lgp_barrier/wall_seconds/minavgmax) are the harness's
   here; the per-request gather loop is untouched bale source. */
double ig_agp(int64_t *tgt, int64_t *index, int64_t l_num_req, int64_t *table) {
  int64_t i;
  for (i = 0; i < l_num_req; i++) {
    tgt[i] = lgp_get_int64(table, index[i]); /* <-- verbatim bale */
  }
  return 0;
}

#ifndef L_TBL_SIZE
#define L_TBL_SIZE 100000   /* table elements per PE */
#endif
#ifndef L_NUM_REQ
#define L_NUM_REQ 1000000   /* gather requests per PE */
#endif
#ifndef REPS
#define REPS 1
#endif

int main(void) {
  start_pes(0);
  int me = shmem_my_pe();
  int np = shmem_n_pes();
  int64_t l_tbl_size = L_TBL_SIZE;
  int64_t l_num_req = L_NUM_REQ;
  int64_t tab_siz = (int64_t)l_tbl_size * np;

  int64_t *table = (int64_t *)shmem_malloc((size_t)l_tbl_size * sizeof(int64_t));
  for (int64_t i = 0; i < l_tbl_size; i++)
    table[i] = (-1) * (i * np + me + 1); /* bale ig.upc:152 */

  int64_t *index = (int64_t *)malloc((size_t)l_num_req * sizeof(int64_t));
  int64_t *tgt = (int64_t *)malloc((size_t)l_num_req * sizeof(int64_t));
  unsigned long st = 0x9e3779b97f4a7c15UL ^ ((unsigned long)me * 0xff51afd7ed558ccdUL + 1);
  for (int64_t i = 0; i < l_num_req; i++) {
    st = st * 6364136223846793005UL + 1442695040888963407UL;
    index[i] = (int64_t)((st >> 17) % (unsigned long)tab_siz);
  }

  shmem_barrier_all();
  struct timeval t0, t1;
  gettimeofday(&t0, 0);
  for (int r = 0; r < REPS; r++)
    ig_agp(tgt, index, l_num_req, table);
  gettimeofday(&t1, 0);
  double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1e6;

  /* bale correctness: tgt[i] == -(index[i]+1) == table[index[i]] */
  int64_t errors = 0, checksum = 0;
  for (int64_t i = 0; i < l_num_req; i++) {
    if (tgt[i] != (-1) * (index[i] + 1))
      errors++;
    checksum += tgt[i] * (1 + (i % 131));
  }
  static long part_err, part_sum;
  part_err = errors;
  part_sum = checksum;
  shmem_barrier_all();
  if (me == 0) {
    long terr = 0, tsum = 0, x;
    for (int p = 0; p < np; p++) {
      shmem_long_get(&x, &part_err, 1, p); terr += x;
      shmem_long_get(&x, &part_sum, 1, p); tsum += x;
    }
    printf("Begining Kernel 1 execution.\n");
    printf("BALE indexgather (ig_agp) np=%d l_tbl_size=%ld l_num_req=%ld reps=%d errors=%ld checksum=%ld\n",
           np, l_tbl_size, l_num_req, REPS, terr, tsum);
    printf("\tElapsed time: 0 hour(s), 0 minute(s), %d second(s), %d milliseconds, %d micro second(s).\n",
           (int)sec, (int)((sec - (int)sec) * 1000),
           (int)(((sec * 1000) - (long)(sec * 1000)) * 1000));
  }
  free(index); free(tgt);
  return 0;
}

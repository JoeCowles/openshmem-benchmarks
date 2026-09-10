/* zhalo.c — faithful, self-contained reproduction of tmLQCD's naive OpenSHMEM
 * Z-direction halo exchange (src/lib/xchange/xchange_field.c, TM_USE_SHMEM path).
 *
 * The real code (lines 305-330) does, per lattice site:
 *   for (k = 0; k < T*LX*LY/2; k++) {
 *     shmem_double_put((double*)(l + x0), (double*)(l + g_field_z_ipt_even[k]), 24, g_nb_z_dn);
 *     x0++;
 *   }
 * i.e. T*LX*LY/2 separate 24-double (one spinor) puts to the SAME neighbor PE,
 * with a contiguous/affine DEST (l+x0, x0++) and a GATHERED source (l+idx[k]).
 * The MPI path packs the same boundary into one MPI_Isend(12*T*LX*LY) — the
 * message-aggregation opportunity this diagnostic exercises.
 *
 * This TU strips tmLQCD's autoconf/MPI/geometry dependencies but preserves the
 * exact op the pass sees: a loop of shmem_double_put with nelems=24, invariant
 * PE, affine dest, indirect source. Build knobs (compile-time so the loop bound
 * is a plain counted loop the pass can analyze; the real dims are runtime EXTERN
 * ints, which the pass also handles):
 *   -DNSITES=<n>   number of boundary sites (loop trip count)   default 4096
 */
#include <shmem.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NSITES
#define NSITES 4096
#endif

#define SPINOR_DOUBLES 24   /* one spinor = 12 complex = 24 doubles (tmLQCD) */

int main(void) {
  shmem_init();
  int me = shmem_my_pe();
  int npes = shmem_n_pes();
  int nbr_dn = (me - 1 + npes) % npes;   /* g_nb_z_dn analogue: one fixed neighbor */

  long ndoub = (long)(NSITES + NSITES) * SPINOR_DOUBLES; /* interior + halo slack */

  /* symmetric spinor field `l`, flattened to doubles */
  double *l = (double *)shmem_malloc(sizeof(double) * ndoub);
  /* precomputed gather map g_field_z_ipt_even[k] (indirect source offsets) */
  int *idx = (int *)malloc(sizeof(int) * NSITES);
  for (int k = 0; k < NSITES; k++) idx[k] = (k * 7 % NSITES) * SPINOR_DOUBLES; /* scattered */
  for (long i = 0; i < ndoub; i++) l[i] = (double)(me * 1000 + i);

  shmem_barrier_all();

  /* ---- the naive tmLQCD Z-halo: one 24-double put per site to ONE neighbor ---- */
  int x0 = NSITES * SPINOR_DOUBLES;                 /* contiguous dest base (halo region) */
  for (int k = 0; k < NSITES; k++) {
    shmem_double_put((double *)(l + x0), (double *)(l + idx[k]), SPINOR_DOUBLES, nbr_dn);
    x0 += SPINOR_DOUBLES;
  }

  shmem_barrier_all();

  /* touch the result so nothing is dead-code-eliminated */
  double acc = 0.0;
  for (long i = NSITES * SPINOR_DOUBLES; i < ndoub; i++) acc += l[i];
  if (me == 0) printf("zhalo done: NSITES=%d npes=%d acc=%g\n", NSITES, npes, acc);

  shmem_free(l);
  free(idx);
  shmem_finalize();
  return 0;
}

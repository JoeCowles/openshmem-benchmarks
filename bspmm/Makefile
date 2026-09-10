# -*- Mode: Makefile; -*-
CC=gcc 
CFLAGS= -Wall -O2 -I/home/bmichalowicz/2025_Summer_Internship/utils/include
OPENMPFLAGS=-fopenmp -lmpi
LDFLAGS=-Wl,-rpath,/home/bmichalowicz/2025_Summer_Internship/utils/lib -L/home/bmichalowicz/2025_Summer_Internship/utils/lib -lshmem
BSPMM_COMMON_SRC=bspmm_common.c
BINS=osh_original
BINS+=osh_debug
BINS+=bspmm_single
BINS+=bspmm_multiple
BINS+=bspmm_multiple_nwins
BINS+=bspmm_single_debug
BINS+=original
BINS+=og_debug
BINS+=osh_nb
BINS+=osh_nb_debug
BINS+=sec_osh_nb
BINS+=sec_osh_nb_debug

all: $(BINS)

og_nb: bspmm_single_nb.c $(BSPMM_COMMON_SRC)
	mpicc $(LDFLAGS) $(CFLAGS) -O3 $^ -o $@

og_nb_debug: bspmm_single_nb.c $(BSPMM_COMMON_SRC)
	mpicc $(LDFLAGS) $(CFLAGS) -g3 -Og -D DEBUG $^ -o $@

# og_bspmm_single_modified.c
original: bspmm_single.c $(BSPMM_COMMON_SRC)
	mpicc $(LDFLAGS) $(CFLAGS) -O3 $^ -o $@

osh_original: oshmem_bspmm_single.c $(BSPMM_COMMON_SRC)
	oshcc $(LDFLAGS) $(CFLAGS) -O3 $^ -o $@

osh_nb: oshmem_bspmm_single_nb.c $(BSPMM_COMMON_SRC)
	oshcc $(LDFLAGS) $(CFLAGS) -O3 $^ -o $@

osh_nb_debug: oshmem_bspmm_single_nb.c $(BSPMM_COMMON_SRC)
	oshcc $(LDFLAGS) $(CFLAGS) -g -Og -D DEBUG $^ -o $@

osh_debug: oshmem_bspmm_single.c $(BSPMM_COMMON_SRC)
	oshcc $(LDFLAGS) $(CFLAGS) -g -D DEBUG $^ -o $@

og_debug: bspmm_single.c $(BSPMM_COMMON_SRC)
	mpicc $(LDFLAGS) $(CFLAGS) -g3 -Og -D DEBUG $^ -o $@

bspmm_single: bspmm_single_modified.c $(BSPMM_COMMON_SRC)
	$(CC) $(LDFLAGS) $(CFLAGS) -O3 $^ -o $@

debug: bspmm_single_debug

bspmm_single_debug:  bspmm_single_modified.c $(BSPMM_COMMON_SRC)
	$(CC) $(LDFLAGS) $(CFLAGS) -g3 -D DEBUG $^ -o $@

bspmm_multiple: bspmm_multiple.c $(BSPMM_COMMON_SRC)
	$(CC) $(LDFLAGS) $(CFLAGS) -lmpi $(OPENMPFLAGS) $^ -o $@

bspmm_multiple_nwins: bspmm_multiple_nwins.c $(BSPMM_COMMON_SRC)
	$(CC) $(LDFLAGS) $(CFLAGS) -lmpi $(OPENMPFLAGS) $^ -o $@


sec_osh_nb: secure_oshmem_bspmm_single_nb.c $(BSPMM_COMMON_SRC)
	oshcc $(LDFLAGS) $(CFLAGS) -O3 $^ -o $@

sec_osh_nb_debug: secure_oshmem_bspmm_single_nb.c $(BSPMM_COMMON_SRC)
	oshcc $(LDFLAGS) $(CFLAGS) -g -Og -D DEBUG $^ -o $@


clean:
	rm -f $(BINS)



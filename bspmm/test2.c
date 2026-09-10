#include <shmem.h>
#include <stdlib.h>
#include <stdio.h>


int main(int argc, char **argv){

    shmem_init();

    int *buf = shmem_malloc(sizeof(int));
    int *buf2 = shmem_malloc(sizeof(int));

    *buf = 2;
    int val = 2;

    int rank = shmem_my_pe();
int ret = 0;
    if (rank == 1){ 
        ret = shmem_int_atomic_fetch_add(buf, val, 0);
  }
    printf("Rank %d], value: %d, ret: %d\n", shmem_my_pe(), *buf, ret);

    if (rank == 1){ 
        ret = shmem_int_atomic_fetch_add(buf, val, 0);
    }
    printf("Rank %d], value: %d, ret:%d\n", shmem_my_pe(), *buf, ret );

    shmem_free(buf);
    shmem_free(buf2);

    shmem_finalize();
}

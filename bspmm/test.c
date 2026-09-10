#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>


int main(int argc, char **argv){

    int rank;
    int *next, *next_next;

    shmem_init();

    next = shmem_malloc(sizeof(int));
    next_next = shmem_malloc(sizeof(int));

    *next = 0;
    *next_next = 0;
    int cur_op = 0;

    int max = 10;
    shmem_barrier_all();
//    int ret_work_id;
    *next = shmem_int_atomic_fetch_add(next, 2, 0);
    *next_next = *next + 1;
    rank = shmem_my_pe();
    while(*next < max){
       // fprintf(stderr, "rank: %d], next_next: %d, next: %d, max: %d (line: %d)\n", 
       //         rank, *next_next, *next, max, __LINE__);

        cur_op = *next;
        *next = *next_next;

        if(*next < max){
            *next_next = shmem_int_atomic_fetch_add(next_next, 1, 0);
        }
       // fprintf(stderr, "rank: %d], next_next: %d, next: %d, max: %d (line: %d)\n", 
       //         rank, *next_next, *next, max, __LINE__);
    }
    shmem_barrier_all();

    shmem_free(next);
    shmem_free(next_next);

    shmem_finalize();

}
        

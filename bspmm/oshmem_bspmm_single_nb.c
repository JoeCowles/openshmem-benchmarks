/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
#include <sys/time.h>
#include "bspmm.h"
#include <shmem.h>
/*
 * Block sparse matrix multiplication using RMA operations, a global counter for workload
 * distribution, MPI_THREAD_SINGLE mode.
 *
 * A, B, and C denote submatrices (tile_dim x tile_dim) and n is tile_num
 *
 * | C11 ... C1n |   | A11 ... A1n |    | B11 ... B1n |
 * |  . .     .  |   |  . .     .  |    |  . .     .  |
 * |  .  Cij  .  | = |  .  Aik  .  | *  |  .  Bkj  .  |
 * |  .     . .  |   |  .     . .  |    |  .     . .  |
 * | Cn1 ... Cnn |   | An1 ... Ann |    | Bn1 ... Bnn |
 *
 * bspmm parallelizes all indpendent relevant work units. It maitains a table of
 * work units. Each work unit corresponds to 1 DGEMM of tiles (one A tile,
 * one B tile, and one C tile). The non-zero tiles of A, B, and C matrices are
 * evenly distributed amongst the ranks. Each rank will locally accumulate C until
 * its next work unit corresponds to a different C tile.
 *
 * The distribution of work between the ranks of all the workers is dynamic:
 * each rank reads a counter to obtain its work id. The counter is updated
 * atomically each time it is read.
 */

#define OFI_WINDOW_HINTS 0
#define COMPUTE 1
#define FINE_TIME 1
#define WARMUP 1
#define CHECK_FOR_ERRORS 1
#define SHOW_WORKLOAD_DIST 0
static long lock1 = 0,
            lock2 = 0;

static inline long long shmem_offset_of_tile(int global_tile_id, int tot_ranks, int tile_dim)
{
    long long target_offset;
    int rank = shmem_my_pe(); 
#ifdef DEBUG
    fprintf(stderr, "Rank %d], global_tile_id: %d, tot_ranks: %d, tile_dim: %d, ", 
            rank, global_tile_id, tot_ranks, tile_dim);
#endif
    target_offset = (global_tile_id / tot_ranks) * tile_dim * tile_dim;
#ifdef DEBUG
    fprintf(stderr, "Rank %d] target_offset: %lld\n", rank, target_offset);
#endif
    return target_offset;
}

static inline double shmem_getwtime(){
    double wtime = 0.0;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    wtime = tv.tv_sec;
    wtime += (double) tv.tv_usec/1.0e6;
    return wtime;
}

int invoke_get(int nprocs, int rank, size_t tile_dim, int *work_unit_table, int work_id, int work_unit_disp,
                double *sub_mat, double *local_buf, long long disp, double *win, int is_warmup, int *target_rank){
    int i, j;
    int is_mpi_get = 0;
    double *target_tile = NULL;
    int elements_in_tile = tile_dim * tile_dim;
    int global_tile = work_unit_table[work_id * 3 + work_unit_disp];
    (*target_rank) = target_rank_of_tile(global_tile, nprocs);
    long long target_offset = offset_of_tile(global_tile, nprocs, tile_dim);

    if ((*target_rank) == rank) {
        //t_start = START_FINE_TIME(is_warmup);
        target_tile = &sub_mat[(int) target_offset];
        for (i = 0; i < tile_dim; i++) {
            for (j = 0; j < tile_dim; j++) {
                local_buf[i * tile_dim + j] = target_tile[i * tile_dim + j];
            }
        }
        //t_local_get += GET_FINE_TIME(t_start, is_warmup);
        //local_get_counter = INCREMENT_COUNTER(local_get_counter, is_warmup);
    } else {
        //t_start = START_FINE_TIME(is_warmup);
        shmem_double_get(local_buf, win+disp+target_offset, elements_in_tile, *target_rank);        
        //t_get += GET_FINE_TIME(t_start, is_warmup);
        //get_counter = INCREMENT_COUNTER(get_counter, is_warmup);
        is_mpi_get = 1;
    }
    return is_mpi_get;
}

int bspmm_get(int nprocs, int rank, size_t tile_dim, int *work_unit_table, int work_id, double *sub_mat_a,
                double *sub_mat_b, double *local_a, double *local_b, long long disp_a, long long disp_b,
                double *win, int is_warmup, int *target_rank_a, int *target_rank_b) {
    int is_mpi_get_a = 0, is_mpi_get_b = 0, ret = 0;
    is_mpi_get_a = invoke_get(nprocs, rank, tile_dim, work_unit_table, work_id, 0, sub_mat_a, local_a, disp_a, win,
                                is_warmup, target_rank_a);
    is_mpi_get_b = invoke_get(nprocs, rank, tile_dim, work_unit_table, work_id, 1, sub_mat_b, local_b, disp_b, win,
                                is_warmup, target_rank_b);
    if (is_mpi_get_a) {
        ret = 1;
    } else if (is_mpi_get_b) {
        ret = 2;
    } else if (is_mpi_get_a && is_mpi_get_b) {
        ret = 3;
    }
    return ret;
}


#if COMPUTE
void dgemm(double *local_a, double *local_b, double *local_c, int tile_dim);
#endif

int main(int argc, char **argv)
{
    int rank, nprocs;
#if SHOW_WORKLOAD_DIST
    int my_work_counter, *all_worker_counter;
    int *alt_rank;
#endif
    int tile_dim, tile_num, *tile_map;
    size_t elements_in_tile, tile_size;
    size_t sub_mat_elements;
    int tot_non_zero_tiles, tot_tiles_per_rank;
    int p_dim, node_dim;
    int ppn;
    int *work_unit_table, work_units;
    double *sub_mat_a, *sub_mat_b, *sub_mat_c;
//    MPI_Aint disp_a, disp_b, disp_c;
    long long disp_a, disp_b, disp_c;
       
#if OFI_WINDOW_HINTS
    MPI_Info win_info;
#endif

    int work_id = 0, next_work_id = 0,
        temp_work_id = 0;
    int i, k, j;
    int prev_tile_c;
    double *target_tile;
    double *local_a, *local_b, *local_c;
    const int one = 1;
    const int two = 2;
    
    double *win_mem;
    int *counter_win_mem;
    double *win;
    int *win_counter;
    int target_rank_a, target_rank_b;

    int p = 0;


#if FINE_TIME
   /* double t_start;
    double t_get, t_accum;
    double t_get_flush, t_accum_flush;
    double t_per_get, t_per_accum;
    double t_per_get_flush, t_per_accum_flush;
    int get_counter, accum_counter;
    double *t_get_procs, *t_accum_procs;
    double *t_get_flush_procs, *t_accum_flush_procs;
    double min_t_get, max_t_get, mean_t_get;
    double min_t_accum, max_t_accum, mean_t_accum;
    double min_t_get_flush, max_t_get_flush, mean_t_get_flush;
    double min_t_accum_flush, max_t_accum_flush, mean_t_accum_flush;*/
      
#endif
    
    p = 0;
    int in_node_p_dim;
    int node_i, node_j;
    int in_node_i, in_node_j;
    int rank_in_parray;
    //MPI_Comm comm_world;
    shmem_team_t comm_world;

    /* initialize MPI environment */
    shmem_init();
    //MPI_Init(&argc, &argv);
    rank = shmem_my_pe();    //    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    nprocs = shmem_n_pes(); //MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    double *temp_buf[nprocs];

  
#if FINE_TIME
  

   // *total_get_start = 
   //     *total_get_end =
   //     *total_accum_start = 
   //     *total_accum_end =
   //     *get_start_1 = 
   //     *get_start_2 =
   //     *get_end_1 =
   //     *get_end_2 =
   //     *accum_start_1 = 
   //     *accum_start_2 =
   //     *accum_end_1 = 
   //     *accum_end_2 = 0;
   

    int *get_counter = shmem_malloc(sizeof(double)),
        *accum_counter = shmem_malloc(sizeof(double));
    *get_counter = 0;
    *accum_counter = 0;

    int *tot_get_count = shmem_malloc(sizeof(int));
    int *tot_accum_count = shmem_malloc(sizeof(int));;  
   
    double *total_get_start = shmem_malloc(sizeof(double));
    double *total_get_end = shmem_malloc(sizeof(double));
    double *total_accum_start = shmem_malloc(sizeof(double));
    double *total_accum_end = shmem_malloc(sizeof(double));

    double *accum_start_1 = shmem_malloc(sizeof(double)),
           *accum_start_2 = shmem_malloc(sizeof(double)),
           *accum_end_1=shmem_malloc(sizeof(double)) ,
           *accum_end_2 = shmem_malloc(sizeof(double)) ;
    double *get_start_1 = shmem_malloc(sizeof(double)),
           *get_end_1 = shmem_malloc(sizeof(double)),
           *get_start_2 = shmem_malloc(sizeof(double)),
           *get_end_2 = shmem_malloc(sizeof(double));
    double *min_get= shmem_malloc(sizeof(double)),
           *max_get= shmem_malloc(sizeof(double)),
           *avg_get= shmem_malloc(sizeof(double)),
           *min_acc= shmem_malloc(sizeof(double)),
           *max_acc= shmem_malloc(sizeof(double)),
           *avg_acc= shmem_malloc(sizeof(double));
    double *min_dgemm= shmem_malloc(sizeof(double)),
           *max_dgemm= shmem_malloc(sizeof(double)),
           *avg_dgemm= shmem_malloc(sizeof(double));
    double *dgemm_start= shmem_malloc(sizeof(double)), *dgemm_end= shmem_malloc(sizeof(double));
    double *t1 = shmem_malloc(sizeof(double)), *t2 = shmem_malloc(sizeof(double));
    *t1 = 0;
    *t2  = 0;
    *total_accum_start = 0;
    *total_accum_end = 0;
    *accum_start_1 = 0;
     *accum_start_2 = 0; 
     *accum_end_1 = 0;
     *accum_end_2 = 0;
     *get_start_1 = 0;
     *get_start_2 = 0; 
     *get_end_1 = 0;
     *get_start_2 = 0; 
     *min_get = 0; 
     *max_get = 0; 
     *avg_get = 0;
     *min_acc = 0; *max_acc = 0; *avg_acc = 0;
     *min_dgemm = 0; *max_dgemm = 0; *avg_dgemm = 0;
     *dgemm_start = 0; *dgemm_end = 0;
     *total_get_start = 0;
     *total_get_end = 0;

#endif
    

    /* argument checking and setting */
    if (setup(rank, nprocs, argc, argv, &tile_dim, &tile_num, &p_dim, &node_dim, &ppn)) {
        shmem_finalize();  //MPI_Finalize();
        exit(0);
    }
    elements_in_tile = tile_dim * tile_dim;
    tile_size = elements_in_tile * sizeof(double);
    for (p = 0; p<nprocs; p++){
        temp_buf[p] = shmem_malloc(2*(elements_in_tile)*sizeof(double));
    }

    in_node_p_dim = p_dim / node_dim;
    /* find my rank in the processor array from my rank in COMM_WORLD */
    node_i = rank / (ppn * node_dim);
    node_j = (rank / ppn) % node_dim;
    in_node_i = (rank / in_node_p_dim) % in_node_p_dim;
    in_node_j = rank % in_node_p_dim;
    rank_in_parray = (node_i * ppn * node_dim) + (in_node_i * p_dim) + (node_j * in_node_p_dim) + in_node_j;

    /* Change rank to match the logical ranks used in this application */
    /* TODO: How to best do this in OSHMEM with team_split_strided/etc.? */
    //MPI_Comm_split(MPI_COMM_WORLD, 0, rank_in_parray, &comm_world);
    //MPI_Comm_rank(comm_world, &rank);

#if DEBUG
    if (rank == 0) {
        printf("tile_dim %d\n", tile_dim);
        printf("tile_num %d\n", tile_num);
        printf("p_dim %d\n", p_dim);
    }
#endif

    /* Create a map of non-sparse tiles in the whole matrix */
    tile_map = calloc(tile_num * tile_num, sizeof(int));
    init_tile_map(tile_map, tile_num, &tot_non_zero_tiles);
    /* For now, this map is the same for matrices A and B */
 
    /* The non-zero tiles are distributed amongst the ranks in a round-robin fashion */
    tot_tiles_per_rank = tot_non_zero_tiles / nprocs;
    if (tot_non_zero_tiles % nprocs) {
        /* Although only some of the ranks need extra tiles, we allocate the extra tiles
         * to all the ranks so that the displacement from the start of the window's memory
         * to the A (disp_a), B (disp_b) or C (disp_c) submatrix is the same for all ranks.*/ 
        tot_tiles_per_rank++;
    }
#if DEBUG
    printf("Rank %d: Non-zero tiles with me %d\n", rank, tot_tiles_per_rank);
#endif

    /* init work unit table */
    init_work_unit_table(tile_map, tile_map, tile_map, tile_num, &work_unit_table, &work_units);
#if DEBUG
    if (rank == 0) printf("work_units %d\n", work_units);
#endif

#if OFI_WINDOW_HINTS
 
#endif

    /* Non-zero tiles distributed evenly between the processes */
    sub_mat_elements = elements_in_tile * tot_tiles_per_rank;

    /* Allocate and create RMA windows for the tiles in A, B, and C */
    win = shmem_malloc(3*sub_mat_elements*sizeof(double));
    win_mem = win;


    sub_mat_a = win_mem;
    sub_mat_b = sub_mat_a + sub_mat_elements;
    sub_mat_c = sub_mat_b + sub_mat_elements;
    shmem_set_lock(&lock1);
    init_sub_mats(sub_mat_a, sub_mat_b, sub_mat_c, sub_mat_elements);
    shmem_clear_lock(&lock1);
  

    /* Allocate RMA window for the counter that allows for load balancing */

    win_counter = shmem_malloc(sizeof(int));
    counter_win_mem = win_counter;


    disp_a = 0;
    disp_b = disp_a + sub_mat_elements;
    disp_c = disp_b + sub_mat_elements;
    int disp_d = disp_a;
    int disp_e = disp_b;

    shmem_barrier_all(); //MPI_Barrier(comm_world);

 

    local_a = shmem_calloc(elements_in_tile, sizeof(double));
    local_b = shmem_calloc(elements_in_tile, sizeof(double));
    local_c = shmem_calloc(elements_in_tile, sizeof(double));

    /* For separate buffers */
    double *local_d = shmem_calloc(elements_in_tile, sizeof(double));
    double *local_e = shmem_calloc(elements_in_tile, sizeof(double));

    double *base_local_a = local_a,
           *base_local_b = local_b,
           *base_local_c = local_c,
           *base_local_d = local_d,
           *base_local_e = local_e; 

    double *next_local_a = NULL,
           *next_local_b = NULL,
           *next_local_d = NULL,
           *next_local_e = NULL;
    int cur_buf_idx = 0,
        next_buf_idx = 0;
    int num_iters = 0;

   shmem_barrier_all(); // MPI_Barrier(MPI_COMM_WORLD);
    /* Benchmark */

    prev_tile_c = -1; 
#if SHOW_WORKLOAD_DIST
    my_work_counter = 0;
#endif

    shmem_barrier_all(); // MPI_Barrier(comm_world);
#if FINE_TIME
   
    //t_get = t_accum = 0;
   // t_get_flush = t_accum_flush = 0;
    *get_counter =0 ;
    *accum_counter = 0;
#else
    *t1 = shmem_getwtime();
#endif

    shmem_barrier_all(); //MPI_Barrier(comm_world);
    *t1 = shmem_getwtime();
    /* Framework: 
     * Get a
     * Get b
     *
     * while() {
     *      getnb(c)
     *      getnb(d)
     *      dgemm()
     *      wait(c, d)
     *      memcpy(a, c, bytes)
     *      memcpy(b, d, bytes)
     *}
     */

   // shmem_set_lock(&lock1);
    work_id = shmem_int_atomic_fetch_add(counter_win_mem, two, 0);
  //  shmem_clear_lock(&lock1);


  //  shmem_set_lock(&lock1);
    int global_tile_a = work_unit_table[work_id * 3 + 0];
    target_rank_a = target_rank_of_tile(global_tile_a, nprocs);
    long long target_offset_a = shmem_offset_of_tile(global_tile_a, nprocs, tile_dim);
 //   shmem_clear_lock(&lock1);

    /* Obtain tile A */
    if (target_rank_a == rank) {
        /* Copy tile A from local memory */
        shmem_set_lock(&lock1);
        target_tile = &sub_mat_a[(int) target_offset_a];
        for (i = 0; i < tile_dim; i++)
            for (k = 0; k < tile_dim; k++)
                local_a[i*tile_dim + k] = target_tile[i*tile_dim + k];
        shmem_clear_lock(&lock1);
    } else {
        shmem_fence();
        shmem_set_lock(&lock1);
        shmem_double_get(local_a, win + target_offset_a + disp_a, elements_in_tile, target_rank_a);
        shmem_clear_lock(&lock1);
        shmem_fence();
        shmem_quiet();
    }

  //  shmem_set_lock(&lock2);
    int global_tile_b = work_unit_table[work_id * 3 + 1];
    target_rank_b = target_rank_of_tile(global_tile_b, nprocs);
    long long target_offset_b = shmem_offset_of_tile(global_tile_b, nprocs, tile_dim);
  //  shmem_clear_lock(&lock2);

    if (target_rank_b == rank) {
        /* Copy tile B from local memory */
        shmem_set_lock(&lock1);
        target_tile = &sub_mat_b[(int) target_offset_b];
        for (k = 0; k < tile_dim; k++)
            for (j = 0; j < tile_dim; j++)
                local_b[k*tile_dim + j] = target_tile[k*tile_dim + j];
        shmem_clear_lock(&lock1);
    } else {
        shmem_set_lock(&lock1);
        shmem_double_get(local_b, win+disp_b+target_offset_b, elements_in_tile, target_rank_b);
        shmem_quiet();
        shmem_clear_lock(&lock1);
    }
    shmem_set_lock(&lock1);
    next_work_id = work_id + 1;
    shmem_clear_lock(&lock1);
    fflush(stdout);
    prev_tile_c = -1;

//    fprintf(stderr, "Made it to while loop\n");
    while(work_id < work_units){
        shmem_set_lock(&lock2);
        temp_work_id = work_id;
        work_id = next_work_id;
        shmem_clear_lock(&lock2);

        /* Accumulate time */
        int global_tile_c = work_unit_table[temp_work_id * 3 + 2];
        if (global_tile_c != prev_tile_c && prev_tile_c >= 0) {
            /* MPI_Accumulate locally accumulated C before proceeding */
            shmem_set_lock(&lock1);
            int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
            long long target_offset_c = shmem_offset_of_tile(prev_tile_c, nprocs, tile_dim);
            shmem_clear_lock(&lock1);
#if DEBUG
            double tile_sum = 0;
            int tile_i, tile_j;
            for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                    tile_sum += local_c[tile_i*tile_dim + tile_j];
                }
            }
            printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %lld\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c); 
#endif
            /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
#if FINE_TIME
            *accum_counter = (*accum_counter) + 1;
            //t_start = shmem_getwtime();
#endif

            shmem_set_lock(&lock2);
         //   shmem_fence();
            *accum_start_1 = shmem_getwtime();
            *total_accum_start = shmem_getwtime();
            shmem_double_get((temp_buf[rank]), win + disp_c + target_offset_c, elements_in_tile, target_rank_c);
         //   shmem_fence();
            shmem_quiet();
            for (p = 0; p<elements_in_tile; p++){
                local_c[p] += temp_buf[rank][p];
            }
         //   shmem_fence();
            shmem_double_put((win + disp_c + target_offset_c), local_c, elements_in_tile, target_rank_c);
            *accum_end_1 = *accum_end_1 + (shmem_getwtime() - *accum_start_1);
            *total_accum_end = *total_accum_end +  (shmem_getwtime() - *total_accum_start);
         //   shmem_fence();
            shmem_quiet();
            shmem_clear_lock(&lock2);
            memset(local_c, 0, tile_size);
        }
        prev_tile_c = global_tile_c;
  //      local_d = base_local_d + (cur_buf_idx * elements_in_tile);
  //      local_e = base_local_e + (cur_buf_idx * elements_in_tile);
  //      next_buf_idx = (cur_buf_idx + 1) % 2;
  //      next_local_e = base_local_e + (next_buf_idx * elements_in_tile);
  //      next_local_d = base_local_d + (next_buf_idx * elements_in_tile);
  //      cur_buf_idx += next_buf_idx;

        if (work_id < work_units){
#if FINE_TIME
            (*get_counter) = *(get_counter)+1;
#endif
            shmem_set_lock(&lock1);
            int global_tile_d = work_unit_table[work_id * 3 + 0];
            int target_rank_d = target_rank_of_tile(global_tile_d, nprocs);
            long long target_offset_d = shmem_offset_of_tile(global_tile_d, nprocs, tile_dim);
            shmem_clear_lock(&lock1);

            /* Obtain tile A */

            *get_start_1 = shmem_getwtime();
            *total_get_start = shmem_getwtime();
            if (target_rank_d == rank) {
                /* Copy tile A from local memory */
                shmem_set_lock(&lock1);
                target_tile = &sub_mat_a[(int) target_offset_d];
                for (i = 0; i < tile_dim; i++)
                    for (k = 0; k < tile_dim; k++)
                        local_d[i*tile_dim + k] = target_tile[i*tile_dim + k];
                shmem_clear_lock(&lock1);
            } else {
                //shmem_fence();
                shmem_set_lock(&lock1);
#if DEBUG
                if (num_iters == 0){
                    fprintf(stdout, "[Rank %d] [get_nbi_1] Message size (elements in tile * sizeof(double) = %ld\n",
                            rank,
                            sizeof(double) * elements_in_tile);
                }
          
                printf("Rank %d] local_e: Get size: %ld\n", rank, elements_in_tile *sizeof(double));
#endif
                shmem_double_get_nbi(local_d, win + target_offset_d + disp_d, elements_in_tile, target_rank_d);
                
                shmem_clear_lock(&lock1);
                //shmem_fence();
                //shmem_quiet();
            }
            *get_end_1 = *get_end_1 +  (shmem_getwtime() - *get_start_1);
            *total_get_end = *total_get_end + (shmem_getwtime() - *total_get_start);

            shmem_set_lock(&lock1);
            int global_tile_e = work_unit_table[work_id * 3 + 1];
            int target_rank_e = target_rank_of_tile(global_tile_e, nprocs);
            long long target_offset_e = shmem_offset_of_tile(global_tile_e, nprocs, tile_dim);
            shmem_clear_lock(&lock1);

            /* Obtain tile B */
            *total_get_start = shmem_getwtime();
            *get_start_2 = shmem_getwtime();
            if (target_rank_e == rank) {
                /* Copy tile A from local memory */
                shmem_set_lock(&lock1);
                target_tile = &sub_mat_b[(int) target_offset_e];
                for (i = 0; i < tile_dim; i++)
                    for (k = 0; k < tile_dim; k++)
                        local_e[i*tile_dim + k] = target_tile[i*tile_dim + k];
                shmem_clear_lock(&lock1);
            } else {
                shmem_fence();
                shmem_set_lock(&lock1);
#if DEBUG
                if (num_iters == 0){
                    fprintf(stdout, "[Rank %d] [get_nbi_2] Message size (elements in tile * sizeof(double) = %ld\n",
                            rank,
                            sizeof(double) * elements_in_tile);
                }

                printf("Rank %d] local_e: Get size: %ld\n", rank, elements_in_tile *sizeof(double)); 
#endif
                shmem_double_get_nbi(local_e, win + target_offset_e + disp_e, elements_in_tile, target_rank_e);
                
                shmem_clear_lock(&lock1);
                shmem_fence();
                shmem_quiet();
            }
            
            *get_end_2 = *get_end_2 +  (shmem_getwtime() - *get_start_2);
            *total_get_end = *total_get_end +  (shmem_getwtime() - *total_get_start);
        }

        *dgemm_start = shmem_getwtime();
        dgemm(local_a, local_b, local_c, tile_dim);  
 //       double temp = shmem_getwtime()- * dgemm_start;
        *dgemm_end = *dgemm_end + (shmem_getwtime()-*dgemm_start); 
        shmem_quiet();
       

       // shmem_set_lock(&lock1);
        next_work_id = shmem_int_atomic_fetch_add(counter_win_mem, one,0);
      //  shmem_clear_lock(&lock1);
        if (work_id < work_units){
            double *temp;
            temp = local_a;
            local_a = local_d;
            local_d = temp;
            temp = local_b;
            local_b = local_e;
            local_e = temp;
           // memcpy(local_a, local_d, elements_in_tile * sizeof(double));
           // memcpy(local_b, local_e, elements_in_tile * sizeof(double));
        }
        num_iters++;
    }

    

    if (prev_tile_c >= 0) {
        /* MPI_Accumulate locally accumulated C before finishing */
      //  shmem_set_lock(&lock2);
        int target_rank_c = target_rank_of_tile(prev_tile_c, nprocs);
        long long target_offset_c = shmem_offset_of_tile(prev_tile_c, nprocs, tile_dim);
      //  shmem_clear_lock(&lock2);
        
        /* accumulate tile C (always use MPI since we need to ensure atomicity during accumulation) */
#if DEBUG
        double tile_sum = 0;
        int tile_i, tile_j;
        for (tile_i = 0; tile_i < tile_dim; tile_i++) {
            for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                tile_sum += local_c[tile_i*tile_dim + tile_j];
            }
        }
        printf("Rank %d accumulating tile %d with value %.1f on rank %d using offset %lld\n", rank, prev_tile_c, tile_sum, target_rank_c, target_offset_c); 
#endif
#if FINE_TIME
        //accum_counter++;
        //t_start = shmem_getwtime();
#endif  
        shmem_set_lock(&lock2);
    //    shmem_fence();
    //    *accum_start_2 = shmem_getwtime();
    //    *total_accum_start = shmem_getwtime();
        shmem_double_get((temp_buf[rank]), win + disp_c + target_offset_c, elements_in_tile, target_rank_c);
        shmem_quiet();
        for (p = 0; p<elements_in_tile; p++){
            local_c[p] += temp_buf[rank][p];
        }
      //  shmem_fence();
        shmem_double_put((win + disp_c + target_offset_c), local_c, elements_in_tile, target_rank_c);
       // *accum_end_2 += (shmem_getwtime() - *accum_start_2);
       // *total_accum_end += (shmem_getwtime() - *total_accum_start);
        shmem_quiet();
        shmem_clear_lock(&lock2);

        //MPI_Accumulate(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile,
          //         MPI_DOUBLE, MPI_SUM, win);
#if FINE_TIME
        //t_accum += (shmem_getwtime() - t_start);
       // t_start = shmem_getwtime();
#endif
       // MPI_Win_flush(target_rank_c, win);
#if FINE_TIME
        //t_accum_flush += (shmem_getwtime() - t_start);
#endif
    }
 
    shmem_barrier_all();
    *t2 = shmem_getwtime(); 
    shmem_barrier_all();
#if DEBUG
    fprintf(stderr,"Rank %d] num_iters: %d\n", rank, num_iters);
#endif
    shmem_barrier_all();
    //printf("Rank %d: sub_mat_c[0] is %.1f\n", rank, sub_mat_c[0]);
    //printf("Rank %d: sub_mat_c[1] is %.1f\n", rank, sub_mat_c[1]);
    //printf("Rank %d done!\n", rank);

#if FINE_TIME
#if 0
    if (get_counter > 0) { 
        t_per_get = t_get / get_counter;
        t_per_get_flush = t_get_flush / get_counter;
    } else
        t_per_get = t_per_get_flush = 0;

    if (accum_counter > 0) {
        t_per_accum = t_accum / accum_counter;
        t_per_accum_flush = t_accum_flush / accum_counter;
    } else
        t_per_accum = t_per_accum_flush = 0;

    if (rank == 0) {
        t_get_procs = calloc(nprocs, sizeof(double));
        t_accum_procs = calloc(nprocs, sizeof(double));
        t_get_flush_procs = calloc(nprocs, sizeof(double));
        t_accum_flush_procs = calloc(nprocs, sizeof(double));
        if (!t_get_procs || !t_accum_procs || !t_get_flush_procs || !t_accum_flush_procs) {
            fprintf(stderr, "Unable to allocate memory for t_get_procs, t_get_flush_procs, t_accum_flush_procs, or t_accum_procs\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else
 //       t_get_procs = t_accum_procs = t_get_flush_procs = t_accum_flush_procs = NULL;
  //  MPI_Gather(&t_per_get, 1, MPI_DOUBLE, t_get_procs, 1, MPI_DOUBLE, 0, comm_world);
  //  MPI_Gather(&t_per_get_flush, 1, MPI_DOUBLE, t_get_flush_procs, 1, MPI_DOUBLE, 0, comm_world);
  //  MPI_Gather(&t_per_accum, 1, MPI_DOUBLE, t_accum_procs, 1, MPI_DOUBLE, 0, comm_world);
  //  MPI_Gather(&t_per_accum_flush, 1, MPI_DOUBLE, t_accum_flush_procs, 1, MPI_DOUBLE, 0, comm_world);
#endif
    *tot_get_count = 0;
    *tot_accum_count = 0;
    shmem_int_sum_reduce(SHMEM_TEAM_WORLD, tot_get_count, get_counter, 1);
    shmem_int_sum_reduce(SHMEM_TEAM_WORLD, tot_accum_count, accum_counter, 1);
//    MPI_Reduce(&get_counter, &tot_get_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//    MPI_Reduce(&accum_counter, &tot_accum_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#if 0
    
 //   min_t_get = max_t_get = mean_t_get = 0;
//    min_t_accum = max_t_accum = mean_t_accum = 0;
 //   min_t_get_flush = max_t_get_flush = mean_t_get_flush = 0;
//    min_t_accum_flush = max_t_accum_flush = mean_t_accum_flush = 0;

    if (rank == 0) {
        int pi;
        double sum_t_get, sum_t_accum, sum_t_get_flush, sum_t_accum_flush;
        int nworkers_who_got, nworkers_who_accumed;

        nworkers_who_got = nworkers_who_accumed = 0;

 //       max_t_get = max_t_accum = max_t_get_flush = max_t_accum_flush = -1;
//        min_t_get = min_t_accum = min_t_get_flush = min_t_accum_flush = 9999;
//        sum_t_get = sum_t_accum = sum_t_get_flush = sum_t_accum_flush = 0;

        for (pi = 0; pi < nprocs; pi++) {
            if (t_get_procs[pi] > 0) {
                nworkers_who_got++;
                if (max_t_get < t_get_procs[pi])
                    max_t_get = t_get_procs[pi];
                if (min_t_get > t_get_procs[pi])
                    min_t_get = t_get_procs[pi];
                sum_t_get += t_get_procs[pi];
                /* Who got also flushed */
                if (max_t_get_flush < t_get_flush_procs[pi])
                    max_t_get_flush = t_get_flush_procs[pi];
                if (min_t_get_flush > t_get_flush_procs[pi])
                    min_t_get_flush = t_get_flush_procs[pi];
                sum_t_get_flush += t_get_flush_procs[pi];
            }
            
            if (t_accum_procs[pi] > 0) { 
                nworkers_who_accumed++;
                if (max_t_accum < t_accum_procs[pi])
                    max_t_accum = t_accum_procs[pi];
                if (min_t_accum > t_accum_procs[pi])
                    min_t_accum = t_accum_procs[pi];
                sum_t_accum += t_accum_procs[pi];
                /* Who accumed also flushed */
                if (max_t_accum_flush < t_accum_flush_procs[pi])
                    max_t_accum_flush = t_accum_flush_procs[pi];
                if (min_t_accum_flush > t_accum_flush_procs[pi])
                    min_t_accum_flush = t_accum_flush_procs[pi];
                sum_t_accum_flush += t_accum_flush_procs[pi];
            }
        }

        mean_t_get = sum_t_get / nworkers_who_got;
        mean_t_accum = sum_t_accum / nworkers_who_accumed;
        mean_t_get_flush = sum_t_get_flush / nworkers_who_got;
        mean_t_accum_flush = sum_t_accum_flush / nworkers_who_accumed;
    
#endif /*if 0*/
#else
    }
    *t2 = shmem_getwtime();

     
#endif
    shmem_barrier_all();
#if DEBUG
    fprintf(stderr, "Rank %d] get_counter: %d\n", rank, *get_counter);

    if (rank == 0 ){
        fprintf(stderr, "tot_get_count: %d\n", *tot_get_count);
    }
#endif
    shmem_barrier_all();

    *(total_get_end) = (*total_get_end) / (double)(*get_counter); 
    shmem_double_max_reduce(SHMEM_TEAM_WORLD, max_get, total_get_end, 1);  
    //    MPI_Reduce(&total_get_end, &max_get, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    shmem_double_min_reduce(SHMEM_TEAM_WORLD, min_get, total_get_end, 1); 
    //MPI_Reduce(&total_get_end, &min_get, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    shmem_double_sum_reduce(SHMEM_TEAM_WORLD, avg_get, total_get_end, 1); 
          //&total_get_end, &avg_get, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    *(avg_get) = *avg_get/nprocs;

    *(dgemm_end) = (*dgemm_end) / (double)num_iters;
    
    shmem_double_max_reduce(SHMEM_TEAM_WORLD, max_dgemm, dgemm_end, 1);
    shmem_double_min_reduce(SHMEM_TEAM_WORLD, min_dgemm, dgemm_end, 1);
    shmem_double_sum_reduce(SHMEM_TEAM_WORLD, avg_dgemm, dgemm_end, 1);

    //MPI_Reduce(&dgemm_end, &max_dgemm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&dgemm_end, &min_dgemm, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&dgemm_end, &avg_dgemm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    *(avg_dgemm) = (*avg_dgemm)/nprocs;

    (*total_accum_end) =  (*total_accum_end) / (double)(*accum_counter);

    shmem_double_max_reduce(SHMEM_TEAM_WORLD, max_acc, total_accum_end, 1);
    shmem_double_min_reduce(SHMEM_TEAM_WORLD, min_acc, total_accum_end, 1);
    shmem_double_sum_reduce(SHMEM_TEAM_WORLD, avg_acc, total_accum_end, 1);

    //MPI_Reduce(&total_accum_end, &max_acc, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&total_accum_end, &min_acc, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    //MPI_Reduce(&total_accum_end, &avg_acc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    *avg_acc = (*avg_acc)/ (double)nprocs;

    
#if SHOW_WORKLOAD_DIST
    if (rank == 0) {
        all_worker_counter = calloc(nprocs, sizeof(int));
        alt_rank = calloc(nprocs, sizeof(int));
    } else {
        all_worker_counter = NULL;
        alt_rank = NULL;
    }

    MPI_Gather(&my_work_counter, 1, MPI_INT, all_worker_counter, 1, MPI_INT, 0, comm_world);
    MPI_Gather(&rank, 1, MPI_INT, alt_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif

   // MPI_Win_sync(win);    
   /* MEM_MODE: synchronize private and public window copies */
    if (rank == 0) {
        int mat_dim = tile_num * tile_dim;
#if SHOW_WORKLOAD_DIST
        printf("Worker\tReal-rank\tUnits\n");
        for (i = 0; i < nprocs; i++) {
            printf("%d\t%d\t%d\n", i, alt_rank[i], all_worker_counter[i]);
        }
        printf("\n");
#endif
#if FINE_TIME
        /* Each of the times reported are per operation */
    //    printf("mat_dim,tile_dim,work_units,nworkers,"
     //           "min_get_time,max_get_time,mean_get_time,"
      //          "min_accum_time,max_accum_time,mean_accum_time,"
      //          "min_get_flush_time,max_get_flush_time,mean_get_flush_time,"
      //          "min_accum_flush_time,max_accum_flush_time,mean_accum_flush_time\n");
   //     printf("%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,\n", mat_dim, tile_dim, work_units, nprocs,
    //            min_t_get, max_t_get, mean_t_get,
    //            min_t_accum, max_t_accum, mean_t_accum,
    //            min_t_get_flush, max_t_get_flush, mean_t_get_flush,
    //            min_t_accum_flush, max_t_accum_flush, mean_t_accum_flush);
        
        printf("TOTAL_TIME,MIN_COMP,MAX_COMP,AVG_COMP,MIN_GET,MAX_GET,AVG_GET,MIN_ACC,MAX_ACC,AVG_ACC\n");
        printf("%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,\n", 
               (*t2 - *t1)/num_iters*1e6,
               (*min_dgemm)*1e6, (*max_dgemm)*1e6, (*avg_dgemm)*1e6, 
               (*min_get)*1e6, (*max_get)*1e6, (*avg_get)*1e6,
               (*min_acc)*1e6, (*max_acc)*1e6, (*avg_acc) *1e6);

        printf("mat_dim, tile_dim, work_units, num_workers\n");
        printf("%d, %d, %d, %d\n",
                mat_dim, tile_dim, work_units, nprocs);
#else
        /* This time is the time for the whole kernel i.e. the observed time by the end user of the application */
        printf("mat_dim,tile_dim,work_units,nworkers,time\n");
        printf("%d,%d,%d,%d,%.9f\n", mat_dim, tile_dim, work_units, nprocs, t2 - t1);
#endif
    }

#if CHECK_FOR_ERRORS
    if (rank == 0) {
        /* Check matrices */
        size_t mat_dim = tile_num * tile_dim;
        double *mat_a = calloc(mat_dim * mat_dim, sizeof(double));
        double *mat_b = calloc(mat_dim * mat_dim, sizeof(double));
        double *mat_correct_c = calloc(mat_dim * mat_dim, sizeof(double));
        double *mat_c = calloc(mat_dim * mat_dim, sizeof(double));
        
        init_mat_according_to_map(mat_a, mat_dim); 
        init_mat_according_to_map(mat_b, mat_dim); 

        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
               mat_c[i*mat_dim + j] = 0; 
               mat_correct_c[i*mat_dim + j] = 0; 
            }
        }

        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                for (k = 0; k < mat_dim; k++) {
                    mat_correct_c[i*mat_dim + j] += mat_a[i*mat_dim + k] * mat_b[k*mat_dim + j];
                }
            }
        }
        int tile_i, tile_j;
        for (i = 0; i < tile_num; i++) {
            for (j = 0; j < tile_num; j++) {
                shmem_set_lock(&lock1);
                int global_tile_c = tile_map[i*tile_num + j];
                shmem_clear_lock(&lock1);
                if (global_tile_c != -1) {
                    shmem_set_lock(&lock1);
                    int target_rank_c = target_rank_of_tile(global_tile_c, nprocs);
                    long long target_offset_c = shmem_offset_of_tile(global_tile_c, nprocs, tile_dim); 
                    shmem_clear_lock(&lock1);
                    shmem_fence();
                    shmem_double_get(local_c, win+disp_c+target_offset_c, elements_in_tile, target_rank_c);
                    shmem_fence();
                    shmem_quiet();
//                    MPI_Get(local_c, elements_in_tile, MPI_DOUBLE, target_rank_c, disp_c + target_offset_c, elements_in_tile, MPI_DOUBLE, win);
//                    MPI_Win_flush(target_rank_c, win);
#if DEBUG
                    //printf("Rank %d got tile %d with value %.1f from rank %d using offset %d\n", rank, global_tile_c, local_c[0], target_rank_c, target_offset_c); 
#endif
                    for (tile_i = 0; tile_i < tile_dim; tile_i++) {
                        for (tile_j = 0; tile_j < tile_dim; tile_j++) {
                         //   shmem_set_lock(&lock1);
                           mat_c[i*tile_dim*mat_dim + j*tile_dim + tile_i*mat_dim + tile_j] = local_c[tile_i*tile_dim + tile_j];      //   shmem_clear_lock(&lock1);
                        }
                    }
                }
            }
        }
        /* Check for errors */
        int errors = 0;
#if DEBUG
        printf("Correct matrix:\n");
        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                printf("%.1f\t", mat_correct_c[i*mat_dim + j]); 
            }
            printf("\n");
        }
        printf("\n");
        printf("Computed matrix:\n");
        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                printf("%.1f\t", mat_c[i*mat_dim + j]); 
            }
            printf("\n");
        }
#endif
        for (i = 0; i < mat_dim; i++) {
            for (j = 0; j < mat_dim; j++) {
                if (mat_correct_c[i*mat_dim + j] != mat_c[i*mat_dim + j])
                    errors++;
            }
        }
        if (errors)
            fprintf(stdout, "Found %d errors\n", errors);
        if (errors == 0)
            fprintf(stdout, "Test passed!\n");

        free(mat_a);
        free(mat_b);
        free(mat_correct_c);
        free(mat_c);
    }
#endif

 //   MPI_Win_unlock_all(win);
//    MPI_Win_unlock_all(win_counter);

//    MPI_Win_free(&win_counter);
//    MPI_Win_free(&win);
//    MPI_Comm_free(&comm_world);

#if OFI_WINDOW_HINTS
//    MPI_Info_free(&win_info);
#endif    
 
   // shmem_free(local_a);
   // shmem_free(local_b);
   // shmem_free(local_c);

#if FINE_TIME
 
#endif

    free(tile_map);
    free(work_unit_table);

    shmem_free(accum_start_1);
    shmem_free(accum_start_2);
    shmem_free(accum_end_1);
    shmem_free(accum_end_2);
    shmem_free(get_start_1);
    shmem_free(get_start_2);
    shmem_free(get_end_1);
    shmem_free(get_end_2);
    shmem_free(min_get);
    shmem_free(max_get);
    shmem_free(avg_get);
    shmem_free(min_acc);
    shmem_free(max_acc);
    shmem_free(avg_acc);
    shmem_free(max_dgemm);
    shmem_free(min_dgemm);
    shmem_free(avg_dgemm);
    shmem_free(t1);
    shmem_free(t2);
    shmem_free(dgemm_start);
    shmem_free(dgemm_end);
    

    shmem_finalize(); //MPI_Finalize();
    return 0;
}

#if COMPUTE
void dgemm(double *local_a, double *local_b, double *local_c, int tile_dim)
{
    int i, j, k;

    for (j = 0; j < tile_dim; j++) {
        for (i = 0; i < tile_dim; i++) {
            for (k = 0; k < tile_dim; k++)
                local_c[j + i * tile_dim] += local_a[k + i * tile_dim] * local_b[j + k * tile_dim];
        }
    }
}
#endif

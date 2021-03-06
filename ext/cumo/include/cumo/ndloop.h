#ifndef CUMO_NDLOOP_H
#define CUMO_NDLOOP_H

typedef struct NA_LOOP_ITER {
    ssize_t    pos; // - required for each dimension.
    ssize_t    step;
    size_t    *idx;
} cumo_na_loop_iter_t;

typedef struct NA_LOOP_ARGS {
    VALUE    value;
    ssize_t  elmsz;      // element size in bytes, e.g., 4 for int, 8 for double
    char    *ptr;
    //char    *buf_ptr;  //
    int      ndim;       // required for each argument.
    // ssize_t pos; - not required here.
    size_t  *shape;
    cumo_na_loop_iter_t *iter;  // moved from cumo_na_loop_t
} cumo_na_loop_args_t;

// pass this structure to user iterator
typedef struct NA_LOOP {
    int  narg;
    int  ndim;             // n of user dimention used at user function.
    size_t *n;             // n of elements for each dim (=shape)
    cumo_na_loop_args_t *args;  // for each arg
    VALUE  option;
    void  *opt_ptr;
    VALUE  err_type;
    int    reduce_dim;     // number of dimensions to reduce in reduction kernel, e.g., for an array of shape: [2,3,4],
                           // 3 for sum(), 1 for sum(axis: 1), 2 for sum(axis: [1,2])
    VALUE  reduce;         // dimension indicies to reduce in reduction kernel (in bits), e.g., for an array of shape:
                           // [2,3,4], 111b for sum(), 010b for sum(axis: 1), 110b for sum(axis: [1,2])
} cumo_na_loop_t;


// ------------------ ndfunc -------------------------------------------

#define NDF_HAS_LOOP            (1<<0) // x[i]
#define NDF_STRIDE_LOOP         (1<<1) // *(x+stride*i)
#define NDF_INDEX_LOOP          (1<<2) // *(x+idx[i])
#define NDF_KEEP_DIM            (1<<3)
#define NDF_INPLACE             (1<<4)
#define NDF_ACCEPT_BYTESWAP     (1<<5)

#define NDF_FLAT_REDUCE         (1<<6)
#define NDF_EXTRACT             (1<<7)
#define NDF_CUM                 (1<<8)

#define NDF_INDEXER_LOOP        (1<<9) // Cumo custom. Use cumo own indexer.

#define FULL_LOOP       (NDF_HAS_LOOP|NDF_STRIDE_LOOP|NDF_INDEX_LOOP|NDF_INPLACE)
#define FULL_LOOP_NIP   (NDF_HAS_LOOP|NDF_STRIDE_LOOP|NDF_INDEX_LOOP)
#define STRIDE_LOOP     (NDF_HAS_LOOP|NDF_STRIDE_LOOP|NDF_INPLACE)
#define STRIDE_LOOP_NIP (NDF_HAS_LOOP|NDF_STRIDE_LOOP)
#define NO_LOOP         0

#define OVERWRITE Qtrue // used for CASTABLE(t)

#define NDF_TEST(nf,fl)  ((nf)->flag & (fl))
#define NDF_SET(nf,fl)  {(nf)->flag |= (fl);}

#define NDF_ARG_READ_ONLY   1
#define NDF_ARG_WRITE_ONLY  2
#define NDF_ARG_READ_WRITE  3

// type of user function
typedef void (*cumo_na_iter_func_t) _((cumo_na_loop_t *const));
typedef VALUE (*cumo_na_text_func_t) _((char *ptr, size_t pos, VALUE opt));
//typedef void (*) void (*loop_func)(cumo_ndfunc_t*, cumo_na_md_loop_t*))


typedef struct NDF_ARG_IN {
    VALUE   type;    // argument types
    int     dim;     // # of dimension of argument handled by user function
                     // if dim==-1, reduce dimension
} cumo_ndfunc_arg_in_t;

typedef struct NDF_ARG_OUT {
    VALUE   type;    // argument types
    int     dim;     // # of dimension of argument handled by user function
    size_t *shape;
} cumo_ndfunc_arg_out_t;

// spec of user function
typedef struct NDFUNCTION {
    cumo_na_iter_func_t func;    // user function
    unsigned int flag;      // what kind of loop user function supports
    int nin;                // # of arguments
    int nout;               // # of results
    cumo_ndfunc_arg_in_t *ain;   // spec of input arguments
    cumo_ndfunc_arg_out_t *aout; // spec of output result
} cumo_ndfunc_t;

#endif /* CUMO_NDLOOP_H */

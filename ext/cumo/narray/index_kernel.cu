#include <cstdint>
#include "cumo/template_kernel.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

__global__ void na_parse_array_index_kernel(size_t* idx, ssize_t* nidxp, ssize_t size, size_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (nidxp[i] < 0) {
            nidxp[i] += size;
        }
        idx[i] = nidxp[i];
    }
}

void na_parse_array_index_kernel_launch(size_t* idx, ssize_t* nidxp, ssize_t size, size_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    na_parse_array_index_kernel<<<gridDim, blockDim>>>(idx,nidxp,size,n);
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

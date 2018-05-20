<% unless c_iter.include? 'robject' %>
__global__ void <%="cumo_#{c_iter}_index_index_kernel"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + idx1[i]) = <%=macro%>(*(<%=dtype%>*)(p2 + idx2[i]));
    }
}

__global__ void <%="cumo_#{c_iter}_stride_index_kernel"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + (i * s1)) = <%=macro%>(*(<%=dtype%>*)(p2 + idx2[i]));
    }
}

__global__ void <%="cumo_#{c_iter}_index_stride_kernel"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + idx1[i]) = <%=macro%>(*(<%=dtype%>*)(p2 + (i * s2)));
    }
}

//<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_stride_stride_kernel_dim#{idim}"%>(na_iarray_t a1, na_iarray_t a2, na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)(p1) = <%=macro%>(*(<%=dtype%>*)(p2));
    }
}
//<% end %>

void <%="cumo_#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_index_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,idx1,idx2,n);
}

void <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,s1,idx2,n);
}

void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_index_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,idx1,s2,n);
}

void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(na_iarray_t* a1, na_iarray_t* a2, na_indexer_t* indexer)
{
    size_t gridDim = get_gridDim(indexer->total_size);
    size_t blockDim = get_blockDim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_stride_stride_kernel_dim#{idim}"%><<<gridDim, blockDim>>>(*a1,*a2,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_stride_stride_kernel_dim"%><<<gridDim, blockDim>>>(*a1,*a2,*indexer);
        break;
    }
}

<% end %>

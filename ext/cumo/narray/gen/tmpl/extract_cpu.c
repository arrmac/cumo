/*
  Extract an element only if self is a dimensionless NArray.
  @overload extract_cpu
  @return [Numeric,Cumo::NArray]
  --- Extract element value as Ruby Object if self is a dimensionless NArray,
  otherwise returns self.
  This method is compatible with Numo NArray's `extract` method.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    volatile VALUE v;
    char *ptr;
    cumo_narray_t *na;
    GetNArray(self,na);

    if (na->ndim==0) {
        ptr = cumo_na_get_pointer_for_read(self) + cumo_na_get_offset(self);
        SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        v = m_extract(ptr);
        cumo_na_release_lock(self);
        return v;
    }
    return self;
}

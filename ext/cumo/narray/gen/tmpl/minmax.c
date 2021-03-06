<% (is_float ? ["","_nan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
    size_t   n;
    char    *p1;
    ssize_t  s1;
    dtype    xmin,xmax;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);

    SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    f_<%=name%><%=j%>(n,p1,s1,&xmin,&xmax);

    *(dtype*)(lp->args[1].ptr + lp->args[1].iter[0].pos) = xmin;
    *(dtype*)(lp->args[2].ptr + lp->args[2].iter[0].pos) = xmax;
}
<% end %>

/*
  <%=name%> of self.
<% if is_float %>
  @overload <%=name%>(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN if exist).
<% else %>
  @overload <%=name%>(axis:nil, keepdims:false)
<% end %>
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::<%=class_name%>,Cumo::<%=class_name%>] min and max of self.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE|NDF_EXTRACT, 2,2, ain,aout};

  <% if is_float %>
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_nan);
  <% else %>
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  <% end %>
    return cumo_na_ndloop(&ndf, 2, self, reduce);
}

void <%="cumo_#{c_func(-1)}_kernel_launch"%>(dtype *ptr, dtype x);

/*
  Array element(s) set.
  @overload []=(dim0,..,dimL,val)
  @param [Numeric,Range,etc] dim0,..,dimL  Multi-dimensional Index.
  @param [Numeric,Cumo::NArray,etc] val  Value(s) to be set to self.
  @return [Numeric] returns val (last argument).

  --- Replace element(s) at +dim0+, +dim1+, ... (index/range/array/true
  for each dimention). Broadcasting mechanism is applied.

  @example
      a = Cumo::DFloat.new(3,4).seq
      => Cumo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [4, 5, 6, 7],
       [8, 9, 10, 11]]

      a[1,2]=99
      a
      => Cumo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [4, 5, 99, 7],
       [8, 9, 10, 11]]

      a[1,[0,2]] = [101,102]
      a
      => Cumo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [101, 5, 102, 7],
       [8, 9, 10, 11]]

      a[1,true]=99
      a
      => Cumo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [99, 99, 99, 99],
       [8, 9, 10, 11]]

*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos;
    char *ptr;
    VALUE a;
    dtype x;

    argc--;
    if (argc==0) {
        <%=c_func.sub(/_aset/,"_store")%>(self, argv[argc]);
    } else {
        nd = na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
        if (nd) {
            a = na_aref_main(argc, argv, self, 0, nd);
            <%=c_func.sub(/_aset/,"_store")%>(a, argv[argc]);
        } else {
            x = <%=type_name%>_extract_data(argv[argc]);
            ptr = na_get_pointer_for_read_write(self) + pos;
            <%="cumo_#{c_func(-1)}_kernel_launch"%>((dtype*)ptr, x);
        }

    }
    return argv[argc];
}

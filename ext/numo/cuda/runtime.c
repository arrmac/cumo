#include <ruby.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "numo/cuda/runtime.h"

VALUE numo_cuda_eRuntimeError;
VALUE numo_cuda_mRuntime;
#define eRuntimeError numo_cuda_eRuntimeError
#define mRuntime numo_cuda_mRuntime

void
numo_cuda_runtime_check_status(cudaError_t status)
{
    if (status != 0) {
        rb_raise(eRuntimeError, "%s (error=%d)", cudaGetErrorString(status), status);
    }
}
#define check_status(status) (numo_cuda_runtime_check_status((status)))

///////////////////////////////////////////
// Initialization
///////////////////////////////////////////

static VALUE
rb_cudaDriverGetVersion(VALUE self)
{
    int _version;
    cudaError_t status;

    status = cudaDriverGetVersion(&_version);

    check_status(status);
    return INT2NUM(_version);
}

static VALUE
rb_cudaRuntimeGetVersion(VALUE self)
{
    int _version;
    cudaError_t status;

    status = cudaRuntimeGetVersion(&_version);

    check_status(status);
    return INT2NUM(_version);
}

/////////////////////////////////////////
// Device and context operations
/////////////////////////////////////////

static VALUE
rb_cudaGetDevice(VALUE self)
{
    int _device;
    cudaError_t status;

    status = cudaGetDevice(&_device);

    check_status(status);
    return INT2NUM(_device);
}

static VALUE
rb_cudaDeviceGetAttributes(VALUE self, VALUE attrib, VALUE device)
{
    int _attrib = NUM2INT(attrib);
    int _device = NUM2INT(device);
    int _ret;
    cudaError_t status;

    status = cudaDeviceGetAttribute(&_ret, _attrib, _device);

    check_status(status);
    return INT2NUM(_ret);
}

static VALUE
rb_cudaGetDeviceCount(VALUE self)
{
    int _count;
    cudaError_t status;

    status = cudaGetDeviceCount(&_count);

    check_status(status);
    return INT2NUM(_count);
}

static VALUE
rb_cudaSetDevice(VALUE self, VALUE device)
{
    int _device = NUM2INT(device);
    cudaError_t status;

    status = cudaSetDevice(_device);

    check_status(status);
    return Qnil;
}

static VALUE
rb_cudaDeviceSynchronize(VALUE self)
{
    cudaError_t status;
    status = cudaDeviceSynchronize();
    check_status(status);
    return Qnil;
}

void
Init_numo_cuda_runtime()
{
    VALUE mNumo = rb_define_module("Numo");
    VALUE mCUDA = rb_define_module_under(mNumo, "CUDA");
    mRuntime = rb_define_module_under(mCUDA, "Runtime");
    eRuntimeError = rb_define_class_under(mCUDA, "RuntimeError", rb_eStandardError);

    rb_define_singleton_method(mRuntime, "cudaDriverGetVersion", rb_cudaDriverGetVersion, 0);
    rb_define_singleton_method(mRuntime, "cudaRuntimeGetVersion", rb_cudaRuntimeGetVersion, 0);
    rb_define_singleton_method(mRuntime, "cudaGetDevice", rb_cudaGetDevice, 0);
    rb_define_singleton_method(mRuntime, "cudaDeviceGetAttributes", rb_cudaDeviceGetAttributes, 2);
    rb_define_singleton_method(mRuntime, "cudaGetDeviceCount", rb_cudaGetDeviceCount, 0);
    rb_define_singleton_method(mRuntime, "cudaSetDevice", rb_cudaSetDevice, 1);
    rb_define_singleton_method(mRuntime, "cudaDeviceSynchronize", rb_cudaDeviceSynchronize, 0);
}
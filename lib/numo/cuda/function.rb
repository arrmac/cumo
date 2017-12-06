require_relative '../cuda'

module Numo::CUDA
  # CUDA kernel function.
  class Function
    # @param [Numo::CUDA::Module] mod
    # @param [String] function name
    def initialize(mod, funcname)
        @module = module  # to keep module loaded
        @ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(@ tuple grid, tuple block, args, size_t shared_mem=0,
                 stream=None):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]
        s = _get_stream(stream)
        _launch(
            @ptr,
            max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
            max(1, block[0]), max(1, block[1]), max(1, block[2]),
            args, shared_mem, s)

    cpdef linear_launch(@ size_t size, args, size_t shared_mem=0,
                        size_t block_max_size=128, stream=None):
        # TODO(beam2d): Tune it
        gridx = size // block_max_size + 1
        if gridx > 65536:
            gridx = 65536
        if size > block_max_size:
            size = block_max_size
        s = _get_stream(stream)
        _launch(@ptr,
                gridx, 1, 1, size, 1, 1, args, shared_mem, s)

add_library(cfd_cuda_backend STATIC
  cuda_backend.cu
  kernels.cu
  euler_backend.cu
  euler_kernels.cu
)

set_target_properties(cfd_cuda_backend PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set_property(TARGET cfd_cuda_backend PROPERTY CUDA_ARCHITECTURES 75)
endif()

target_include_directories(cfd_cuda_backend
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(cfd_cuda_backend PUBLIC cxx_std_20)

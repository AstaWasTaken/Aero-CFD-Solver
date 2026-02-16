set(CFD_CORE_SOURCES
  mesh/airfoil_mesh.cpp
  mesh/mesh.cpp
  numerics/euler_flux.cpp
  numerics/reconstruction.cpp
  numerics/residual_stub.cpp
  io/io_vtk_stub.cpp
  post/forces.cpp
  solvers/euler_solver.cpp
)

add_library(cfd_core_lib ${CFD_CORE_LIBRARY_TYPE} ${CFD_CORE_SOURCES})
add_library(cfd_core ALIAS cfd_core_lib)

set_target_properties(cfd_core_lib PROPERTIES
  OUTPUT_NAME cfd_core_native
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cfd_core_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(cfd_core_lib PUBLIC cxx_std_20)
target_compile_definitions(cfd_core_lib
  PUBLIC
    CFD_CORE_VERSION_STR="${PROJECT_VERSION}"
    $<$<BOOL:${CFD_CUDA_AVAILABLE}>:CFD_HAS_CUDA=1>
    $<$<NOT:$<BOOL:${CFD_CUDA_AVAILABLE}>>:CFD_HAS_CUDA=0>
)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cfd_core_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CFD_CUDA_AVAILABLE)
  add_subdirectory(cuda)
  target_link_libraries(cfd_core_lib PUBLIC cfd_cuda_backend)
  set_target_properties(cfd_core_lib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
endif()

install(TARGETS cfd_core_lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

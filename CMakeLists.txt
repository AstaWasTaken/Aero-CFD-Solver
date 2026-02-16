cmake_minimum_required(VERSION 3.24)

project(
  AeroCFD
  VERSION 0.1.0
  DESCRIPTION "GPU-accelerated CFD scaffold for airfoils and wings"
  LANGUAGES CXX
)

include(GNUInstallDirs)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(CFD_ENABLE_CUDA "Enable CUDA backend" ON)
option(CFD_BUILD_PYTHON "Build pybind11 Python module" ON)
option(CFD_BUILD_TESTS "Build smoke tests" ON)
option(CFD_CORE_SHARED "Build cfd_core as a shared library" OFF)

if(CFD_CORE_SHARED)
  set(CFD_CORE_LIBRARY_TYPE SHARED)
else()
  set(CFD_CORE_LIBRARY_TYPE STATIC)
endif()

set(CFD_CUDA_AVAILABLE OFF)
if(CFD_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CFD_CUDA_AVAILABLE ON)
  else()
    message(WARNING "CFD_ENABLE_CUDA=ON but no CUDA compiler detected; continuing with CPU backend.")
  endif()
endif()

add_subdirectory(src/cfd_core)

add_executable(cfd_native_cli src/apps/cfd_cli.cpp)
target_link_libraries(cfd_native_cli PRIVATE cfd_core_lib)
target_compile_features(cfd_native_cli PRIVATE cxx_std_20)

if(CFD_BUILD_PYTHON)
  add_subdirectory(src/cfd_core/bindings)
endif()

if(CFD_BUILD_TESTS)
  include(CTest)
  if(BUILD_TESTING)
    add_executable(cfd_core_smoke_test tests/cpp/test_smoke.cpp)
    target_link_libraries(cfd_core_smoke_test PRIVATE cfd_core_lib)
    target_compile_features(cfd_core_smoke_test PRIVATE cxx_std_20)
    add_test(NAME cfd_core_smoke COMMAND cfd_core_smoke_test)

    add_executable(cfd_scalar_parity_test tests/cpp/test_scalar_parity.cpp)
    target_link_libraries(cfd_scalar_parity_test PRIVATE cfd_core_lib)
    target_compile_features(cfd_scalar_parity_test PRIVATE cxx_std_20)
    add_test(NAME cfd_scalar_parity COMMAND cfd_scalar_parity_test)

    add_executable(cfd_euler_regression_test tests/cpp/test_euler_regression.cpp)
    target_link_libraries(cfd_euler_regression_test PRIVATE cfd_core_lib)
    target_compile_features(cfd_euler_regression_test PRIVATE cxx_std_20)
    add_test(NAME cfd_euler_regression COMMAND cfd_euler_regression_test)

    add_executable(cfd_euler_parity_test tests/cpp/test_euler_parity.cpp)
    target_link_libraries(cfd_euler_parity_test PRIVATE cfd_core_lib)
    target_compile_features(cfd_euler_parity_test PRIVATE cxx_std_20)
    add_test(NAME cfd_euler_parity COMMAND cfd_euler_parity_test)
  endif()
endif()

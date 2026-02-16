@echo off
setlocal

set "ROOT=%~dp0.."
set "BUILD_DIR=%ROOT%\build\cuda_ninja"

if "%VSCMD_VER%"=="" (
  set "VSDEV_PATH="
  for %%P in (
    "%ProgramFiles%\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
    "%ProgramFiles%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
  ) do (
    if exist %%~P set "VSDEV_PATH=%%~P"
  )
  if "%VSDEV_PATH%"=="" (
    echo Could not find VsDevCmd.bat. Open an x64 Visual Studio Developer prompt and retry.
    exit /b 1
  )
  call "%VSDEV_PATH%" -arch=amd64 -host_arch=amd64 >nul
)

if not defined CUDACXX (
  for /f "usebackq delims=" %%I in (`where nvcc 2^>nul`) do (
    set "CUDACXX=%%I"
    goto :got_nvcc
  )
)
:got_nvcc
if not defined CUDACXX (
  echo nvcc not found in PATH. Install CUDA toolkit or set CUDACXX.
  exit /b 1
)

if not defined CUDAHOSTCXX (
  for /f "usebackq delims=" %%I in (`where cl 2^>nul`) do (
    set "CUDAHOSTCXX=%%I"
    goto :got_cl
  )
)
:got_cl
if not defined CUDAHOSTCXX (
  echo cl not found in PATH. Run from VS Developer prompt.
  exit /b 1
)

if not exist "%ROOT%\build" mkdir "%ROOT%\build"

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER="%CUDACXX%" -DCMAKE_CUDA_HOST_COMPILER="%CUDAHOSTCXX%" -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -DCFD_ENABLE_CUDA=ON -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" -j 8
if errorlevel 1 exit /b 1

ctest --test-dir "%BUILD_DIR%" --output-on-failure
if errorlevel 1 exit /b 1

echo CUDA build and tests completed: %BUILD_DIR%
exit /b 0

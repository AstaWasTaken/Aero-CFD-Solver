@echo off
setlocal EnableExtensions

set "ROOT=%~dp0.."
set "BUILD_DIR=%ROOT%\build\cuda_ninja"

set "VSDEV_PATH="
if defined VSINSTALLDIR (
  if exist "%VSINSTALLDIR%Common7\Tools\VsDevCmd.bat" (
    set "VSDEV_PATH=%VSINSTALLDIR%Common7\Tools\VsDevCmd.bat"
  )
)

if "%VSDEV_PATH%"=="" (
  for %%P in (
    "%ProgramFiles%\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
    "%ProgramFiles%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
  ) do (
    if exist %%~P set "VSDEV_PATH=%%~P"
  )
)

if "%VSDEV_PATH%"=="" (
  if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq delims=" %%I in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
      if exist "%%~I\Common7\Tools\VsDevCmd.bat" (
        set "VSDEV_PATH=%%~I\Common7\Tools\VsDevCmd.bat"
      )
    )
  )
)

if "%VSDEV_PATH%"=="" (
  echo Could not find VsDevCmd.bat. Install Visual Studio C++ build tools and retry.
  exit /b 1
)

call "%VSDEV_PATH%" -arch=amd64 -host_arch=amd64 >nul
if errorlevel 1 (
  echo Failed to initialize Visual Studio x64 developer environment.
  exit /b 1
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
  if defined VCToolsInstallDir (
    if exist "%VCToolsInstallDir%bin\Hostx64\x64\cl.exe" (
      set "CUDAHOSTCXX=%VCToolsInstallDir%bin\Hostx64\x64\cl.exe"
      goto :got_cl
    )
  )
  for /f "usebackq delims=" %%I in (`where cl 2^>nul ^| findstr /I /R "\\Hostx64\\x64\\cl\.exe$"`) do (
    set "CUDAHOSTCXX=%%I"
    goto :got_cl
  )
  for /f "usebackq delims=" %%I in (`where cl 2^>nul`) do (
    set "CUDAHOSTCXX=%%I"
    goto :got_cl
  )
)
:got_cl
if not defined CUDAHOSTCXX (
  echo cl not found in PATH after VsDevCmd initialization.
  exit /b 1
)

if not exist "%CUDAHOSTCXX%" (
  echo CUDA host compiler path does not exist:
  echo   %CUDAHOSTCXX%
  exit /b 1
)

set "HOST_CHECK=%CUDAHOSTCXX:\Hostx64\x64\=%"
if /I "%HOST_CHECK%"=="%CUDAHOSTCXX%" (
  echo CUDA host compiler is not x64:
  echo   %CUDAHOSTCXX%
  echo Expected a path containing Hostx64\\x64\\cl.exe
  exit /b 1
)

if not exist "%ROOT%\build" mkdir "%ROOT%\build"

if exist "%BUILD_DIR%\CMakeCache.txt" (
  findstr /I "\\Hostx86\\x86\\cl.exe" "%BUILD_DIR%\CMakeCache.txt" >nul
  if not errorlevel 1 (
    echo Removing stale x86-configured build directory: %BUILD_DIR%
    rmdir /s /q "%BUILD_DIR%"
  )
)

if exist "%BUILD_DIR%\CMakeCache.txt" if not exist "%BUILD_DIR%\build.ninja" (
  echo Recreating incomplete build directory: %BUILD_DIR%
  rmdir /s /q "%BUILD_DIR%"
)

echo Using CUDA compiler: %CUDACXX%
echo Using MSVC host compiler: %CUDAHOSTCXX%

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_CXX_COMPILER="%CUDAHOSTCXX%" -DCMAKE_CUDA_COMPILER="%CUDACXX%" -DCMAKE_CUDA_HOST_COMPILER="%CUDAHOSTCXX%" -DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler -DCFD_ENABLE_CUDA=ON -DCFD_BUILD_TESTS=ON -DCFD_BUILD_PYTHON=ON
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" -j 8
if errorlevel 1 exit /b 1

ctest --test-dir "%BUILD_DIR%" --output-on-failure
if errorlevel 1 exit /b 1

echo CUDA build and tests completed: %BUILD_DIR%
exit /b 0

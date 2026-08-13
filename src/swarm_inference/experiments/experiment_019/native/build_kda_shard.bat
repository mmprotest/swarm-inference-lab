@echo off
setlocal

if "%~1"=="" (
  echo usage: build_kda_shard.bat OUTPUT_DLL
  exit /b 2
)

set "VSINSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
set "VSWHERE=%VSINSTALLER%\vswhere.exe"
set "VSDIR="
pushd "%VSINSTALLER%"
for /f "usebackq delims=" %%i in (`vswhere.exe -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
popd
if "%VSDIR%"=="" exit /b 3
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul

nvcc -O3 -std=c++17 -arch=sm_120 -allow-unsupported-compiler -Xcompiler=-W3 -shared ^
  -lcudart "%~dp0kda_shard.cu" -o "%~f1"
exit /b %ERRORLEVEL%

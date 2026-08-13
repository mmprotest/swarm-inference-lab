@echo off
setlocal

if "%~1"=="" (
  echo usage: build_grouped_top16.bat OUTPUT_DLL ARCH
  exit /b 2
)
if "%~2"=="" (
  echo usage: build_grouped_top16.bat OUTPUT_DLL ARCH
  exit /b 2
)

set "VSINSTALLER=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
set "VSDIR="
pushd "%VSINSTALLER%"
for /f "usebackq delims=" %%i in (`vswhere.exe -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
popd
if "%VSDIR%"=="" exit /b 3
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul

nvcc -O3 -std=c++17 -arch=%~2 -allow-unsupported-compiler -Xcompiler=-W3 -shared ^
  -lcudart "%~dp0grouped_top16.cu" -o "%~f1"
exit /b %ERRORLEVEL%

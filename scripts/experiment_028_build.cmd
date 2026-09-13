@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b 1
cmake -S native/experiment_028 -B .runtime/experiment-028/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl "-DLLAMA_ROOT=%CD%/.runtime/e027-llama.cpp" "-DLLAMA_BUILD=%CD%/.runtime/experiment-027/build"
if errorlevel 1 exit /b 1
cmake --build .runtime/experiment-028/build --parallel 4
exit /b %errorlevel%

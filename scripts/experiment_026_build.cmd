@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b 1
cmake -S .runtime/e026-llama.cpp -B .runtime/experiment-026/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA=ON -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_OPENSSL=OFF
if errorlevel 1 exit /b 1
cmake --build .runtime/experiment-026/build --config Release --parallel 8 --target llama-server ggml-rpc-server
exit /b %errorlevel%

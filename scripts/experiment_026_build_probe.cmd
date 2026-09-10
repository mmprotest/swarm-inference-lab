@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /MD /DLLAMA_SHARED /DGGML_SHARED /I.runtime/e026-llama.cpp/include /I.runtime/e026-llama.cpp/ggml/include /I.runtime/e026-llama.cpp/vendor native/experiment_026/state_probe.cpp .runtime/experiment-026/build/src/llama.lib .runtime/experiment-026/build/ggml/src/ggml.lib .runtime/experiment-026/build/ggml/src/ggml-base.lib .runtime/experiment-026/build/ggml/src/ggml-rpc/ggml-rpc.lib /Fe:.runtime/experiment-026/build/bin/e026-probe.exe /Fo:.runtime/experiment-026/build/e026-probe.obj
exit /b %errorlevel%

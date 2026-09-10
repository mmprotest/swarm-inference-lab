@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b 1
cl /nologo /O2 /LD native\experiment_026\cache_hash.c /Fo.runtime\experiment-026\cache_hash.obj /Fe.runtime\experiment-026\cache_hash.dll

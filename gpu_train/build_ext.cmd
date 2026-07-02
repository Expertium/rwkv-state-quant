@echo off
REM Build the RWKV_CUDA extension in-place from csrc using the parent venv python + MSVC.
REM Only builds for the RTX 4070 (sm_89 / Ada) to keep nvcc fast.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d C:\Users\Andrew\rwkv-state-quant\gpu_train
set TORCH_CUDA_ARCH_LIST=8.9
set DISTUTILS_USE_SDK=1
set PY=C:\Users\Andrew\rwkv-anki-autoresearch\.venv\Scripts\python.exe
"%PY%" setup.py build_ext --inplace
echo BUILD_EXIT_%ERRORLEVEL%

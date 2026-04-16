@echo off
echo ========================================
echo    TraceRAG v2.0 - Lancement rapide
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python non trouve. Installez Python 3.10+ depuis python.org
    pause
    exit /b 1
)

echo [2/5] Configuration GPU (GTX 1650 CUDA)...
REM Fix OpenBLAS memory allocation crash on Windows
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
REM Force Ollama GPU acceleration (NVIDIA CUDA)
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_GPU_LAYERS=999
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1

echo [3/5] Demarrage d'Ollama avec GPU CUDA (GTX 1650)...
REM Kill any existing Ollama process to avoid port conflicts
taskkill /f /im ollama.exe >nul 2>&1
timeout /t 2 /nobreak >nul
REM Start Ollama in dedicated CMD window with CUDA env vars inherited
start "Ollama-GPU" /min cmd /k "set CUDA_VISIBLE_DEVICES=0 && set OLLAMA_LLM_LIBRARY=cuda_v12 && set OLLAMA_FLASH_ATTENTION=1 && set OPENBLAS_NUM_THREADS=1 && set OMP_NUM_THREADS=1 && ollama serve"
echo     Ollama GPU (cuda_v12) demarre... Attente 6s...
timeout /t 6 /nobreak >nul

echo [4/5] Installation des dependances...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install fastapi uvicorn python-multipart sentence-transformers chromadb PyMuPDF httpx pydantic pydantic-settings python-docx >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Echec de l'installation des dependances
    pause
    exit /b 1
)

echo [5/5] Lancement du serveur TraceRAG...
echo.
echo ========================================
echo   Ouvrez http://localhost:8000
echo   Appuyez sur Ctrl+C pour arreter
echo ========================================
echo.
python main.py

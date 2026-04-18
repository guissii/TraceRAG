@echo off
setlocal
echo ========================================
echo    TraceRAG v2.0 - Lancement rapide
echo ========================================
echo.

cd /d "%~dp0"

echo [1/6] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas trouve dans le PATH.
    echo Installez Python 3.10+ depuis python.org et cochez "Add Python to PATH".
    pause
    exit /b 1
)

echo [2/6] Configuration de l'environnement virtuel...
if not exist "venv" (
    echo Creation de l'environnement virtuel ^(venv^)...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [3/6] Installation des dependances (requirements.txt)...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERREUR: Echec de l'installation des dependances depuis requirements.txt.
    pause
    exit /b 1
)

echo [4/6] Verification de l'acceleration GPU PyTorch...
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [!] PyTorch CUDA non detecte. 
    echo [!] Telechargement de la version optimisee GPU ^(CUDA 12.1^) - env 2.5 Go, veuillez patienter...
    python -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
)

echo [5/6] Demarrage d'Ollama avec GPU...
REM Kill any existing Ollama process to avoid port conflicts
taskkill /f /im ollama.exe >nul 2>&1
timeout /t 2 /nobreak >nul
REM Start Ollama in dedicated CMD window with CUDA env vars inherited
start "Ollama-GPU" /min cmd /k "set CUDA_VISIBLE_DEVICES=0 && set OLLAMA_LLM_LIBRARY=cuda_v12 && set OLLAMA_FLASH_ATTENTION=1 && set OPENBLAS_NUM_THREADS=1 && set OMP_NUM_THREADS=1 && ollama serve"
echo     Attente de l'initialisation d'Ollama...
timeout /t 6 /nobreak >nul

echo [6/6] Telechargement du modele LLM interactif (phi3)...
echo Cette etape peut prendre quelques minutes si le modele n'est pas encore installe.
ollama pull phi3

echo.
echo ========================================
echo Lancement du serveur TraceRAG...
echo   Ouvrez http://localhost:8000
echo   Appuyez sur Ctrl+C pour arreter
echo ========================================
echo.
python main.py

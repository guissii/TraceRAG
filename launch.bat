@echo off
echo ========================================
echo    RAG Explorer - Lancement rapide
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] Verification de Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python non trouve. Installez Python 3.10+ depuis python.org
    pause
    exit /b 1
)

echo [2/3] Installation des dependances...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install fastapi uvicorn python-multipart sentence-transformers chromadb PyMuPDF httpx pydantic >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Echec de l'installation des dependances
    echo Essayez: python -m pip install fastapi uvicorn python-multipart sentence-transformers chromadb PyMuPDF httpx pydantic
    pause
    exit /b 1
)

echo [3/3] Lancement du serveur RAG...
echo.
echo Ouvrez http://localhost:8000 dans votre navigateur
echo Appuyez sur Ctrl+C pour arreter
echo.
python main.py

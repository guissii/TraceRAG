@echo off
cd /d "%~dp0"
echo ========================================================
echo Désinstallation de la version standard (CPU) de PyTorch...
echo ========================================================
python -m pip uninstall -y torch torchvision torchaudio

echo.
echo ========================================================
echo Installation de PyTorch version optimisée GPU (CUDA 12.1)...
echo ========================================================
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo ========================================================
echo Installation terminée ! 
echo Vous pouvez redémarrer votre serveur (launch.bat).
echo ========================================================
pause

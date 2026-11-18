@echo off
REM Lance l'interface de test du modele squelette (double-clic ou depuis n'importe quel dossier).
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo [ERREUR] venv introuvable. Creer l'environnement :
  echo   python -m venv venv
  echo   venv\Scripts\pip install numpy scipy pandas scikit-learn matplotlib
  echo   venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
  pause
  exit /b 1
)
set PORT=%1
if "%PORT%"=="" set PORT=8765
"venv\Scripts\python.exe" -m skeleton.viewer --port %PORT%
pause

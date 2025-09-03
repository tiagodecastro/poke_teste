@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "LOG=%SCRIPT_DIR%start.log"
set "VENV_PY=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "SETUP_PS1=%SCRIPT_DIR%setup.ps1"

echo [i] Pasta do projeto: "%SCRIPT_DIR%"
echo [i] Log: "%LOG%"

REM 0) sanity checks
if not exist "%SETUP_PS1%" (
  echo [x] setup.ps1 nao encontrado em "%SETUP_PS1%" > "%LOG%"
  type "%LOG%"
  echo.
  echo Pressiona uma tecla para sair...
  pause >nul
  exit /b 1
)

REM 1) venv: se nao existir, corre o setup
if not exist "%VENV_PY%" (
  echo [i] venv nao encontrada -> a correr setup.ps1 ... | tee
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SETUP_PS1%" > "%LOG%" 2>&1
  set "RC=%ERRORLEVEL%"
  type "%LOG%"
  if not "%RC%"=="0" (
    echo [x] Setup falhou com RC=%RC%.
    echo Vê o log em: "%LOG%"
    echo.
    echo Pressiona uma tecla para sair...
    pause >nul
    exit /b %RC%
  )
)

REM 2) (opcional) Tesseract: se nao estiver no path padrao, tenta instalar
REM 2) (opcional) Tesseract: se nao estiver disponível, tenta instalar
where tesseract >nul 2>&1
if errorlevel 1 (
  echo [!] Tesseract nao encontrado. A tentar configurar...
  powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SETUP_PS1%" -InstallTesseract > "%LOG%" 2>&1
  type "%LOG%"
)

REM 3) arranca o bot (usa a venv). 
echo [i] A iniciar o bot...
"%VENV_PY%" "%SCRIPT_DIR%bot_pokemon.py" %*
set "RC=%ERRORLEVEL%"
echo.
echo [i] Processo terminou (RC=%RC%). Vê o log em "%LOG%" se algo correu mal.
echo.
echo Pressiona uma tecla para sair...
pause >nul
exit /b %RC%

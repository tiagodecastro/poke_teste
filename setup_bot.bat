@echo off
echo =========================================
echo   Instalador do PokeBot
echo =========================================

REM Verifica se existe Python no PATH
py --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERRO] Python nao encontrado. Instala o Python 3.11+ primeiro.
    pause
    exit /b
)

REM Cria ambiente virtual
if not exist .venv (
    echo [INFO] A criar ambiente virtual...
    py -m venv .venv
)

REM Ativa o ambiente
echo [INFO] Ativando ambiente virtual...
call .venv\Scripts\activate

REM Instala pacotes necessários
echo [INFO] Instalando requisitos...
pip install --upgrade pip
pip install pywin32 mss opencv-python keyboard numpy pytesseract pydirectinput

echo =========================================
echo   Instalação concluída com sucesso!
echo =========================================
pause

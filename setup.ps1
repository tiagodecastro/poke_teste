<#  setup.ps1 — Bootstrap universal para o bot (Windows)
    Uso tipico: duplo-clique em setup.cmd (este script é chamado em Bypass).
    Flags opcionais:
      -RecreateVenv          # recria a venv do zero
      -InstallPython         # tenta instalar Python 3.x via winget se faltar
      -InstallTesseract      # tenta instalar Tesseract via winget se faltar
      -NoSmoke               # não correr smoke tests
#>

[CmdletBinding()]
param(
  [switch]$RecreateVenv = $false,
  [switch]$InstallPython = $true,
  [switch]$InstallTesseract = $true,
  [switch]$NoSmoke = $false
)

$ErrorActionPreference = 'Stop'
$PSStyle.OutputRendering = 'PlainText'

function Info($m){ Write-Host "[i] $m" -ForegroundColor Cyan }
function Ok($m){ Write-Host "[✓] $m" -ForegroundColor Green }
function Warn($m){ Write-Host "[!] $m" -ForegroundColor Yellow }
function Err($m){ Write-Host "[x] $m" -ForegroundColor Red }

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
Info "Pasta do projeto: $Root"

# ---------------- 0) winget ----------------
$HaveWinget = $false
try { $null = Get-Command winget -ErrorAction Stop; $HaveWinget = $true } catch {}
if(-not $HaveWinget){
  Warn "winget não encontrado. Instalação automática de Python/Tesseract pode não estar disponível."
}

# ---------------- 1) Python do sistema ----------------
function Get-Python {
  $cands = @(
    { & py -3 -c "import sys;print(sys.executable)" 2>$null },
    { & py    -c "import sys;print(sys.executable)" 2>$null },
    { & python -c "import sys;print(sys.executable)" 2>$null }
  )
  foreach($c in $cands){ try{ $p = & $c; if($p){ return $p.Trim() } }catch{} }
  return $null
}

$HostPy = Get-Python
if(-not $HostPy -and $InstallPython -and $HaveWinget){
  Info "Python não encontrado — a instalar via winget…"
  try {
    # Preferir 3.11 LTS-ish; se falhar, tenta o metapackage Python.Python.3
    winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
  } catch { Warn "Falhou a instalação Python 3.11, a tentar Python.Python.3"; winget install -e --id Python.Python.3 --accept-package-agreements --accept-source-agreements }
  Start-Sleep -Seconds 5
  $HostPy = Get-Python
}
if(-not $HostPy){
  Err "Python não encontrado. Instala Python x64 (3.10+), marca 'Add to PATH' e volta a executar."
  exit 1
}
Ok "Python do sistema: $HostPy"

# ---------------- 2) venv ----------------
$Venv = Join-Path $Root ".venv"
$Vpy  = Join-Path $Venv "Scripts\python.exe"

if((Test-Path $Venv) -and $RecreateVenv){
  Warn "A remover venv existente: $Venv"
  try { Remove-Item -Recurse -Force $Venv } catch { Start-Sleep -m 300; Remove-Item -Recurse -Force $Venv }
}
if(-not (Test-Path $Vpy)){
  Info "A criar venv em $Venv"
  & $HostPy -m venv $Venv
}
if(-not (Test-Path $Vpy)){
  Err "Falhou a criação da venv."
  exit 1
}
Ok "Venv pronta: $Vpy"

# ---------------- 3) pip básico ----------------
& $Vpy -m pip --version | Out-Null
& $Vpy -m pip install --upgrade pip setuptools wheel

# ---------------- 4) requirements ----------------
$Req = Join-Path $Root "requirements.txt"
if(-not (Test-Path $Req)){
@"
pydantic>=2.7
opencv-python>=4.8
numpy>=1.26
pytesseract>=0.3.10
rapidfuzz>=3.9
keyboard>=0.13.5
mss>=9.0.1
pywin32>=306
"@ | Set-Content -Encoding UTF8 $Req
  Info "Criado requirements.txt base."
}
Info "A instalar dependências… (isto pode demorar)"
& $Vpy -m pip install -r $Req

# ---------------- 5) Tesseract ----------------
function Get-Tesseract {
  $cands = @(
    { & tesseract --version 2>$null; if ($LASTEXITCODE -eq 0) { return (Get-Command tesseract).Source } },
    { $p = "Tesseract-OCR\tesseract.exe"; $d = Join-Path $env:ProgramFiles $p; if (Test-Path $d) { return $d } },
    { $p = "Tesseract-OCR\tesseract.exe"; $d = Join-Path $env:ProgramFiles(x86) $p; if (Test-Path $d) { return $d } }
  )
  foreach($c in $cands){ try{ $t = & $c; if($t){ return $t.Trim() } }catch{} }
  return $null
}
$TessPath = Get-Tesseract
if($TessPath){
  Ok "Tesseract encontrado: $TessPath"
}else{
  Warn "Tesseract não encontrado."
  if($InstallTesseract -and $HaveWinget){
    Info "A instalar Tesseract (UB-Mannheim) via winget…"
    try{
      winget install -e --id UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
      Start-Sleep -Seconds 3
    }catch{
      Warn "Falha na instalação via winget. Instala manualmente e volta a correr o setup."
    }
    $TessPath = Get-Tesseract
    if($TessPath){ Ok "Tesseract instalado: $TessPath" } else { Warn "Ainda não encontro Tesseract." }
  }else{
    Warn "Instala manualmente (UB-Mannheim build) e volta a correr o setup."
  }
}

# ---------------- 6) config.jsonc (com comentários) ----------------
$Cfg = Join-Path $Root "config.jsonc"
if(-not (Test-Path $Cfg)){
@"
// Config do bot (JSONC = JSON com comentários)
{
  "window_title": "PROClient",                  // título da janela do cliente
  "roi_is_absolute": true,                      // true = ROIs em coords absolutas do ecrã

  "battle_template_name": "default",            // nome do template de batalha (templates/battle/default.png)
  "thr_battle": 0.90,                           // limiar do matchTemplate para considerar “batalha”
  "battle_enter_consec": 2,                     // leituras consecutivas para ENTRAR
  "battle_exit_consec": 2,                      // leituras consecutivas para SAIR

  "poll_interval_s": 0.50,                      // ciclo principal (segundos)
  "change_mse_thr": 20.0,                       // pré-filtro: só faz TM/OCR se MSE > thr

  "name_roi": [3420, 234, 360, 36],             // ROI do nome do Pokémon (x,y,w,h)
  "window_roi": [3845, 650, 260, 210],          // ROI de batalha (x,y,w,h)

  "move_base_s": 0.60,                          // duração base de A/D
  "move_jitter_s": 0.06,                        // jitter de A/D
  "key_delay_s": [0.45, 0.95],                  // atraso aleatório entre teclas dos combos
  "encounter_delay_s": [1.0, 2.0],              // atraso antes do primeiro combo
  "wait_between_combos_s": [8.0, 9.0],          // pausa entre COMBO1 e COMBO2
  "post_combo2_verify_s": [1.0, 1.8],           // espera curta antes de verificar fim da batalha
  "recover_s": [0.5, 1.0],                      // pausa após resolver encontro
  "non_target_period_s": [1.8, 2.3],            // intervalo para pressionar '4' em não-alvos

  "special_keywords": ["shiny", "summer"],      // força captura
  "targets": {                                  // alvos (minúsculas) -> combo por alvo
    "wild ferroseed": ["1", "2"]
  }
}
"@ | Set-Content -Encoding UTF8 $Cfg
  Ok "Criado config.jsonc"
}

# ---------------- 7) run_bot.cmd (atalho para arrancar o bot) ----------------
$RunCmd = Join-Path $Root "run_bot.cmd"
if(-not (Test-Path $RunCmd)){
@"
@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%bot_pokemon.py" %*
"@ | Set-Content -Encoding OEM $RunCmd
  Ok "Criado run_bot.cmd"
}

# ---------------- 8) Smoke tests ----------------
if(-not $NoSmoke){
  Info "Smoke tests (imports + Tesseract)…"
  $code = @"
import cv2, numpy as np, pytesseract, win32api, mss, rapidfuzz, keyboard
print('cv2:', cv2.__version__)
print('numpy:', np.__version__)
print('pytesseract:', pytesseract.__version__)
print('tesseract_cmd:', pytesseract.pytesseract.tesseract_cmd)
"@
  & $Vpy - << $code
  Ok "Smoke tests concluídos."
}

Ok "Setup concluído."
Info "Para arrancar o bot agora:"
Write-Host "  run_bot.cmd" -ForegroundColor White

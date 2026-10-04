@echo off
setlocal DisableDelayedExpansion
if "%~1"=="/?" goto help
if "%~1"=="--help" goto help
set "AERO_RUN_SHARDS=%~1"
set "AERO_RUN_NODES=%~2"
set "AERO_RUN_DATASET=%~3"
if not defined AERO_RUN_SHARDS set /p "AERO_RUN_SHARDS=Number of shards: "
if not defined AERO_RUN_NODES set /p "AERO_RUN_NODES=Nodes per shard: "
if not defined AERO_RUN_DATASET set /p "AERO_RUN_DATASET=Dataset CSV path (without quotes): "
set "AERO_RUN_SCRIPT=%~dp0aero\run.ps1"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "try { $options = @{ Shards = [int]$env:AERO_RUN_SHARDS; Nodes = [int]$env:AERO_RUN_NODES; Dataset = $env:AERO_RUN_DATASET; TotalDataSize = [int]::MaxValue; AutoPrepare = $true }; if ($env:AERO_PYTHON) { $options.Python = $env:AERO_PYTHON }; if ($env:AERO_PYTHON_PACKAGES) { $options.PythonPackages = $env:AERO_PYTHON_PACKAGES }; if ($env:AERO_CHECKPOINT) { $options.Checkpoint = $env:AERO_CHECKPOINT }; if ($env:AERO_PREPARE_ONLY -eq '1') { $options.PrepareOnly = $true }; & $env:AERO_RUN_SCRIPT @options; exit 0 } catch { Write-Host $_ -ForegroundColor Red; exit 1 }"
set "AERO_RUN_EXIT=%ERRORLEVEL%"
echo.
if not "%AERO_RUN_EXIT%"=="0" echo AERO failed. See the error above.
if "%~1"=="" pause
exit /b %AERO_RUN_EXIT%
:help
echo Usage: run_aero.bat SHARDS NODES_PER_SHARD "DATASET_CSV_PATH"
echo Example: run_aero.bat 4 4 "C:\Projects\block-emulator\selectedTxs_300K.csv"
echo Run without arguments for interactive prompts.
echo All eligible dataset rows are used. Each run creates a fresh expTestAERO directory.
echo Optional environment variables: AERO_PYTHON, AERO_PYTHON_PACKAGES, AERO_CHECKPOINT.
echo Set AERO_PREPARE_ONLY=1 to prepare without starting nodes.
exit /b 0

# RAG System Boot Script (PowerShell)
# Checks system resources, sets environment, and launches the server.

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  RAG System Boot Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check free RAM
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
Write-Host "`nFree RAM: ${freeGB} GB"

if ($freeGB -lt 4) {
    Write-Host "WARNING: Less than 4 GB free RAM. System may run slowly." -ForegroundColor Yellow
}
if ($freeGB -lt 2) {
    Write-Host "ERROR: Less than 2 GB free RAM. Aborting." -ForegroundColor Red
    exit 1
}

# Set environment variables
$env:OMP_NUM_THREADS = "4"
$env:TOKENIZERS_PARALLELISM = "false"
# Force Python to use UTF-8 for stdio so emoji / arrows / smart quotes in
# logs and warnings don't crash with UnicodeEncodeError on cp1252 consoles.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Write-Host "OMP_NUM_THREADS=$env:OMP_NUM_THREADS"
Write-Host "TOKENIZERS_PARALLELISM=$env:TOKENIZERS_PARALLELISM"
Write-Host "PYTHONUTF8=$env:PYTHONUTF8"

# Activate venv if exists
$venvPath = Join-Path $PSScriptRoot ".." ".venv" "Scripts" "Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "`nActivating virtual environment..."
    & $venvPath
} else {
    Write-Host "`nNo virtual environment found at $venvPath" -ForegroundColor Yellow
}

# Launch server
Write-Host "`nStarting RAG server..." -ForegroundColor Green
$projectRoot = Join-Path $PSScriptRoot ".."
Push-Location $projectRoot
try {
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
} finally {
    Pop-Location
}

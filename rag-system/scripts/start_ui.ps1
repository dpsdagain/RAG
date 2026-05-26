# Chainlit UI Boot Script (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Chainlit UI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Activate venv if exists
$venvPath = "$PSScriptRoot\..\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "`nActivating virtual environment..."
    & $venvPath
} else {
    Write-Host "`nNo virtual environment found at $venvPath" -ForegroundColor Yellow
}

$projectRoot = "$PSScriptRoot\.."
Push-Location $projectRoot

try {
    Write-Host "`nLaunching UI on http://localhost:8001..." -ForegroundColor Green
    chainlit run app/frontend/chainlit_app.py --port 8001 -w
} finally {
    Pop-Location
}

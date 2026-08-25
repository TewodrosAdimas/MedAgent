# PowerShell Launcher for MedAgent FastAPI REST Service

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting MedAgent Clinical REST API Server" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$UvicornExe = ".\venv\Scripts\uvicorn.exe"
& $UvicornExe src.api.server:app --host 127.0.0.1 --port 8000 --reload

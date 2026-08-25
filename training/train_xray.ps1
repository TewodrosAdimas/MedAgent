# PowerShell Automation Script to Train Chest X-Ray Vision Models in MedAgent

param(
    [string]$Model = "densenet",
    [int]$Epochs = 10,
    [int]$BatchSize = 16,
    [double]$LR = 0.0005
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MedAgent: Training Chest X-Ray Model ($Model)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = ".\venv\Scripts\python.exe"

$ArgsList = @(
    "src/models/vision/train_image.py",
    "--model", $Model,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--lr", $LR
)

& $PythonExe $ArgsList

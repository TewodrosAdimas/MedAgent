# PowerShell Automation Script to Train Multimodal Fusion Models in MedAgent

param(
    [string]$Model = "transformer",
    [int]$Epochs = 10,
    [int]$BatchSize = 16,
    [double]$LR = 0.0003,
    [int]$Samples = 500
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MedAgent: Training Multimodal Fusion Model ($Model)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = ".\venv\Scripts\python.exe"

$ArgsList = @(
    "src/models/fusion/train_multimodal.py",
    "--model", $Model,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--lr", $LR,
    "--samples", $Samples
)

& $PythonExe $ArgsList

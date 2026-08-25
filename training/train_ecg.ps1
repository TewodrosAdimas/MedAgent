# PowerShell Automation Script to Train ECG Signal Models in MedAgent

param(
    [string]$Model = "cnn1d",
    [int]$Epochs = 20,
    [int]$BatchSize = 64,
    [double]$LR = 0.001,
    [int]$Subset = 0
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  MedAgent: Training ECG Signal Model ($Model)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = ".\venv\Scripts\python.exe"

$ArgsList = @(
    "src/models/signals/train_signal.py",
    "--model", $Model,
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--lr", $LR
)

if ($Subset -gt 0) {
    $ArgsList += "--subset"
    $ArgsList += $Subset
}

& $PythonExe $ArgsList

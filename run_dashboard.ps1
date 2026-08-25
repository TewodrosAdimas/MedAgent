# PowerShell Launcher for MedAgent Streamlit Clinical Dashboard

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Starting MedAgent Clinical Decision Support Web App" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$StreamlitExe = ".\venv\Scripts\streamlit.exe"
& $StreamlitExe run src/app/streamlit_app.py --server.port 8501

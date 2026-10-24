Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Starting ForensIQ Local Forensic Web Application..." -ForegroundColor Cyan
Write-Host "  Opening your web browser at http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

& ".\venv\Scripts\python.exe" server.py

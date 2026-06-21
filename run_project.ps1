param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$FrontendUrl = "http://127.0.0.1:5173/index.html"

function Assert-PathExists {
    param(
        [string]$Path,
        [string]$Message
    )

    if (-not (Test-Path $Path)) {
        throw $Message
    }
}

Assert-PathExists $Python "Khong tim thay Python trong .venv. Hay chay setup moi truong truoc."
Assert-PathExists $FrontendRoot "Khong tim thay thu muc frontend."

$BackendCommand = @"
Set-Location '$ProjectRoot'
`$env:PYTHONIOENCODING='utf-8'
& '$Python' -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8001
"@

$FrontendCommand = @"
Set-Location '$FrontendRoot'
`$env:PYTHONIOENCODING='utf-8'
& '$Python' -m http.server 5173
"@

Write-Host "Dang khoi dong backend o http://127.0.0.1:8001 ..."
Start-Process powershell.exe -ArgumentList "-NoExit", "-NoProfile", "-Command", $BackendCommand

Write-Host "Dang khoi dong frontend o http://127.0.0.1:5173 ..."
Start-Sleep -Seconds 2
Start-Process powershell.exe -ArgumentList "-NoExit", "-NoProfile", "-Command", $FrontendCommand

if (-not $NoBrowser) {
    Start-Sleep -Seconds 2
    Start-Process $FrontendUrl
}

Write-Host ""
Write-Host "Da khoi dong xong."
Write-Host "Frontend: $FrontendUrl"
Write-Host "Backend : http://127.0.0.1:8001"
Write-Host "Muon tat project thi dong 2 cua so PowerShell backend/frontend."

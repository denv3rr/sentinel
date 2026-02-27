$ErrorActionPreference = "Stop"

$backend = Start-Job -ScriptBlock {
  python -m uvicorn sentinel.main:create_app --factory --reload --host 127.0.0.1 --port 8765
}

$frontend = Start-Job -ScriptBlock {
  Set-Location "apps/frontend"
  npm run dev
}

Write-Host "Backend job: $($backend.Id)"
Write-Host "Frontend job: $($frontend.Id)"
Write-Host "Press Ctrl+C to stop."

try {
  while ($true) {
    Receive-Job -Job $backend -Keep | Out-Host
    Receive-Job -Job $frontend -Keep | Out-Host
    Start-Sleep -Seconds 1
  }
} finally {
  Stop-Job -Job $backend -ErrorAction SilentlyContinue
  Stop-Job -Job $frontend -ErrorAction SilentlyContinue
  Remove-Job -Job $backend -ErrorAction SilentlyContinue
  Remove-Job -Job $frontend -ErrorAction SilentlyContinue
}
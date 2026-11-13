param([string]$PythonExecutable = "python")
$ErrorActionPreference = "Stop"
Push-Location -LiteralPath $PSScriptRoot
try {
    & $PythonExecutable -u code/evaluate_budget_packet.py --out rerun_evidence --threads 8 --limit 30 2>&1 | Tee-Object -FilePath rerun_console.log
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

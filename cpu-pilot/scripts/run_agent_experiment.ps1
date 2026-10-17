param(
    [ValidateSet('measure','agent','check')][string]$Stage = 'check',
    [string]$RuntimeRoot = '',
    [ValidateSet('groq','gemini','openrouter')][string]$Provider = 'groq',
    [string]$Model = 'qwen/qwen3.8-27b',
    [int]$Sources = 1,
    [string]$RunName = 'ava_smoke_v1',
    [string]$Attempt = 'audited_v2'
)
$ErrorActionPreference = 'Stop'
$pilotRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path $pilotRoot -Parent
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path (Split-Path $repoRoot -Parent) 'agent-audio-visual' }
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'RunName must be a simple directory name' }
if ($Attempt -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Attempt must be a simple label' }
if ($Sources -lt 1 -or $Sources -gt 4) { throw 'Frozen integration batch contains one to four sources' }
$outputRoot = Join-Path $pilotRoot "results/$RunName"
if ($Stage -eq 'check') {
    python -m unittest discover -s (Join-Path $pilotRoot 'tests') -p 'test_ava_environment.py' -v
    if ($LASTEXITCODE -ne 0) { throw 'Environment tests failed' }
    python -m unittest discover -s (Join-Path $pilotRoot 'tests') -p 'test_agent_runner.py' -v
} elseif ($Stage -eq 'measure') {
    python (Join-Path $PSScriptRoot 'run_ava_local.py') --runtime-root $RuntimeRoot --output $outputRoot --max-sources $Sources
} else {
    $modelTag = $Model -replace '[^a-zA-Z0-9_-]','_'
    python (Join-Path $PSScriptRoot 'run_evidence_agent.py') --measurements (Join-Path $outputRoot 'measurements.json') --output (Join-Path $outputRoot "agent_${Provider}_${modelTag}_${Attempt}.json") --provider $Provider --model $Model
}
if ($LASTEXITCODE -ne 0) { throw "Experiment stage failed with exit code $LASTEXITCODE" }

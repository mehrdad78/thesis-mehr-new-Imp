# Run AERO against the bundled CSV with isolated configuration and databases.
[CmdletBinding()]
param(
    [string]$Python = '',
    [string]$PythonPackages = '',
    [string]$Checkpoint = 'aero/checkpoints/policy.pt',
    [string]$Dataset = 'selectedTxs_300K.csv',
    [ValidateRange(1, 2147483647)][int]$TotalDataSize = 300000,
    [ValidateRange(1, 2147483647)][int]$EpochBlocks = 10,
    [ValidateRange(1, 2147483647)][int]$InjectSpeed = 2000,
    [ValidateRange(1, 2147483647)][int]$Shards = 4,
    [ValidateRange(1, 2147483647)][int]$Nodes = 4,
    [switch]$PrepareOnly,
    [switch]$AutoPrepare
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}
if (-not $Python) {
    $venvPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
    $bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $venvPython) { $Python = $venvPython }
    elseif ($AutoPrepare -and (Test-Path -LiteralPath $bundledPython)) {
        $Python = $bundledPython
        if (-not $PythonPackages -and (Test-Path -LiteralPath (Join-Path $projectRoot 'tmp/aero-python/torch'))) {
            $PythonPackages = 'tmp/aero-python'
        }
    }
    else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) { $Python = $pythonCommand.Source }
        else { throw 'Specify -Python with the full path to a Python interpreter with PyTorch installed.' }
    }
}
$Python = Resolve-ProjectPath $Python
$Checkpoint = Resolve-ProjectPath $Checkpoint
$Dataset = Resolve-ProjectPath $Dataset
foreach ($requiredFile in @($Python, $Dataset)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "File missing: $requiredFile" }
}
$keys = Join-Path $PSScriptRoot 'keys'
if (-not $AutoPrepare -and -not (Test-Path -LiteralPath (Join-Path $keys 'public.json'))) {
    throw "Generate node keys first: go run -buildvcs=false ./cmd/aero-keygen -shards $Shards -nodes $Nodes"
}
for ($id = 0; -not $AutoPrepare -and $id -le ($Shards * $Nodes); $id++) {
    if (-not (Test-Path -LiteralPath (Join-Path $keys "$id.key"))) { throw "Missing identity key $id" }
}
$ipTablePath = Join-Path $projectRoot 'ipTable.json'
$ips = $null
if (-not $AutoPrepare) { $ips = Get-Content -LiteralPath $ipTablePath -Raw | ConvertFrom-Json }
for ($shard = 0; -not $AutoPrepare -and $shard -lt $Shards; $shard++) {
    for ($node = 0; $node -lt $Nodes; $node++) {
        if (-not $ips."$shard"."$node") { throw "IP table has no address for shard $shard node $node" }
    }
}
if (-not $AutoPrepare -and -not $ips.'2147483647'.'0') { throw 'IP table has no supervisor address' }
$previousPythonPath = $env:PYTHONPATH
$previousGoCache = $env:GOCACHE
$children = @()
Push-Location $projectRoot
try {
    $paths = @($projectRoot)
    if ($PythonPackages) { $paths += Resolve-ProjectPath $PythonPackages }
    if ($previousPythonPath) { $paths += $previousPythonPath }
    $env:PYTHONPATH = $paths -join [IO.Path]::PathSeparator
    & $Python -c 'import torch, numpy'
    if ($LASTEXITCODE -ne 0) { throw 'Python requires PyTorch and NumPy. Install aero/requirements.txt in the selected interpreter.' }
    $env:GOCACHE = Join-Path $projectRoot 'tmp/go-build'
    if ($AutoPrepare -and $Checkpoint -eq (Resolve-ProjectPath 'aero/checkpoints/policy.pt')) {
        if ($Shards -ne 4 -or -not (Test-Path -LiteralPath $Checkpoint)) {
            $Checkpoint = Join-Path $PSScriptRoot "checkpoints/shards-$Shards/policy.pt"
        }
        if (-not (Test-Path -LiteralPath $Checkpoint)) {
            Write-Host "Creating a small development policy for $Shards shards (128 decisions, up to 2048 transactions)."
            Write-Host 'This initial training is for running the implementation; it is not paper-scale training.'
            & $Python -m aero.train --trace $Dataset --output $Checkpoint --shards $Shards --prefix-digits 2 --epoch-size 128 --limit 2048 --steps 128 --batch-size 32 --minibatch 4 --max-moves 16 --update-epochs 2
            if ($LASTEXITCODE -ne 0) { throw 'Initial policy training failed' }
        }
    }
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw "File missing: $Checkpoint" }
    # Fail before starting any nodes if imports or checkpoint dimensions mismatch.
    & $Python -m aero.preflight $Checkpoint $Shards
    if ($LASTEXITCODE -ne 0) { throw 'Python/checkpoint preflight failed' }
    $exe = Join-Path $projectRoot 'tmp/aero/blockEmulator.exe'
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $exe) | Out-Null
    & go build -buildvcs=false -o $exe .
    if ($LASTEXITCODE -ne 0) { throw 'Go build failed' }
    $runName = 'run-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
    $runDir = Join-Path $projectRoot "expTestAERO/$runName"
    $logs = Join-Path $runDir 'process-logs'
    New-Item -ItemType Directory -Path $logs -Force | Out-Null
    if ($AutoPrepare) {
        $keys = Join-Path $runDir 'keys'
        & go run -buildvcs=false ./cmd/aero-keygen -shards $Shards -nodes $Nodes -dir $keys
        if ($LASTEXITCODE -ne 0) { throw 'Node key generation failed' }
        # Hold every chosen port while allocating the table to prevent duplicates.
        $listeners = @()
        $table = [ordered]@{}
        try {
            for ($shard = 0; $shard -le $Shards; $shard++) {
                $members = [ordered]@{}
                $count = $Nodes
                $tableID = "$shard"
                if ($shard -eq $Shards) { $count = 1; $tableID = '2147483647' }
                for ($node = 0; $node -lt $count; $node++) {
                    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
                    $listener.Start()
                    $listeners += $listener
                    $members["$node"] = "127.0.0.1:$($listener.LocalEndpoint.Port)"
                }
                $table[$tableID] = $members
            }
            [IO.File]::WriteAllText((Join-Path $runDir 'ipTable.json'), ($table | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
        }
        finally { foreach ($listener in $listeners) { $listener.Stop() } }
    }
    else { Copy-Item -LiteralPath $ipTablePath -Destination (Join-Path $runDir 'ipTable.json') }
    $config = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'paramsConfig.example.json') -Raw | ConvertFrom-Json
    $config.DatasetFile = $Dataset
    $config.TotalDataSize = $TotalDataSize
    $config.InjectSpeed = $InjectSpeed
    $config.ExpDataRootDir = 'results'
    $config.AERO.Python = $Python
    $config.AERO.Checkpoint = $Checkpoint
    $config.AERO.KeyDir = $keys
    $config.AERO.EpochBlocks = $EpochBlocks
    # Go's JSON decoder needs UTF-8 without the Windows PowerShell 5 BOM.
    $jsonPath = Join-Path $runDir 'paramsConfig.json'
    [IO.File]::WriteAllText($jsonPath, ($config | ConvertTo-Json -Depth 12), [Text.UTF8Encoding]::new($false))
    Write-Host "Prepared run: $runDir"
    if ($TotalDataSize -eq [int]::MaxValue) { Write-Host 'Dataset: all eligible rows until end of file' }
    else { Write-Host "CSV limit: $TotalDataSize" }
    Write-Host "Shards: $Shards; nodes per shard: $Nodes; epoch blocks: $EpochBlocks"
    Write-Host "Dataset: $Dataset"
    Write-Host "Policy: $Checkpoint"
    if ($PrepareOnly) { return }
    for ($shard = 0; $shard -lt $Shards; $shard++) {
        for ($node = 0; $node -lt $Nodes; $node++) {
            $prefix = Join-Path $logs "s${shard}n${node}"
            $process = Start-Process -FilePath $exe -ArgumentList @('-S', $Shards, '-N', $Nodes, '-s', $shard, '-n', $node) -WorkingDirectory $runDir -WindowStyle Hidden -RedirectStandardOutput "$prefix.stdout.log" -RedirectStandardError "$prefix.stderr.log" -PassThru
            $null = $process.Handle
            $children += $process
        }
    }
    $supervisor = Start-Process -FilePath $exe -ArgumentList @('-S', $Shards, '-N', $Nodes, '-c') -WorkingDirectory $runDir -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'supervisor.stdout.log') -RedirectStandardError (Join-Path $logs 'supervisor.stderr.log') -PassThru
    $null = $supervisor.Handle
    $children += $supervisor
    Write-Host "Running. Logs: $logs"
    Write-Host 'Press Ctrl+C to stop the processes started by this script.'
    $lastLine = ''
    $stopping = $false
    while (-not $supervisor.HasExited) {
        Start-Sleep -Seconds 2
        $supervisor.Refresh()
        $supervisorLines = @(Get-Content -LiteralPath (Join-Path $runDir 'results/log/Supervisor.log') -Tail 40 -ErrorAction SilentlyContinue)
        if ($supervisorLines -match 'now sending cstop message to all nodes') { $stopping = $true }
        foreach ($child in $children) {
            $child.Refresh()
            if ($child.HasExited) { $child.WaitForExit() }
            if ($child.HasExited -and $child.ExitCode -ne 0 -and -not $stopping) {
                # CStop may have been written after the snapshot above but before
                # this node exited. Read again at the point of failure detection.
                $shutdownLog = @(Get-Content -LiteralPath (Join-Path $runDir 'results/log/Supervisor.log') -Tail 40 -ErrorAction SilentlyContinue)
                if ($shutdownLog -match 'now sending cstop message to all nodes') { $stopping = $true }
            }
            # Upstream nodes can exit nonzero when peers close during CStop.
            # Once shutdown begins, the supervisor exit determines run success.
            if ($child.HasExited -and $child.ExitCode -ne 0 -and
                (-not $stopping -or $child.Id -eq $supervisor.Id)) {
                throw "Process $($child.Id) failed with exit code $($child.ExitCode); inspect $logs"
            }
        }
        $line = $supervisorLines | Select-Object -Last 1
        if ($line -and $line -ne $lastLine) { Write-Host $line; $lastLine = $line }
    }
    $supervisor.WaitForExit()
    if ($supervisor.ExitCode -ne 0) { throw "Supervisor failed with exit code $($supervisor.ExitCode); inspect $logs" }
    Write-Host "Finished. Results: $(Join-Path $runDir 'results/result')"
}
finally {
    foreach ($child in $children) {
        $child.Refresh()
        if (-not $child.HasExited) { Stop-Process -Id $child.Id -ErrorAction SilentlyContinue }
    }
    $env:PYTHONPATH = $previousPythonPath
    $env:GOCACHE = $previousGoCache
    Pop-Location
}

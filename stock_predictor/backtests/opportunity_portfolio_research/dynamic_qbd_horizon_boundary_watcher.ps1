[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$ParentPid,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [int[]]$BoundaryHorizons,

    [Parameter(Mandatory = $true)]
    [string[]]$ResumeArguments,

    [string]$PythonExecutable = "python",

    [string]$WorkingDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")),

    [ValidateRange(1, 3600)]
    [int]$PollSeconds = 10,

    [switch]$StopParentAtBoundary
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-WatcherEvent {
    param([string]$Name, [hashtable]$Payload = @{})
    $record = @{
        event = $Name
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        parent_pid = $ParentPid
        boundary_horizons = @($BoundaryHorizons | Sort-Object -Unique)
    }
    foreach ($key in $Payload.Keys) { $record[$key] = $Payload[$key] }
    $json = $record | ConvertTo-Json -Depth 8 -Compress
    $path = Join-Path $resolvedOutputRoot "horizon-boundary-watcher.jsonl"
    Add-Content -LiteralPath $path -Value $json -Encoding UTF8
    Write-Host ("[horizon-watcher] " + $json)
}

function Get-ProcessSnapshot {
    @(Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine)
}

function Get-ProcessTreeIds {
    param([int]$Root)
    $snapshot = Get-ProcessSnapshot
    $ids = [System.Collections.Generic.HashSet[int]]::new()
    [void]$ids.Add($Root)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($row in $snapshot) {
            if ($ids.Contains([int]$row.ParentProcessId) -and $ids.Add([int]$row.ProcessId)) {
                $changed = $true
            }
        }
    }
    @($ids | Sort-Object)
}

function Test-HorizonComplete {
    param([int]$Horizon)
    $prefix = Join-Path $resolvedOutputRoot ("checkpoints\H{0:D2}" -f $Horizon)
    $family = "$prefix-family-results.parquet"
    $yearly = "$prefix-yearly-results.parquet"
    $meta = "$prefix.json"
    if (-not ((Test-Path -LiteralPath $family -PathType Leaf) -and
              (Test-Path -LiteralPath $yearly -PathType Leaf) -and
              (Test-Path -LiteralPath $meta -PathType Leaf))) {
        return $false
    }
    try {
        $payload = Get-Content -LiteralPath $meta -Raw -Encoding UTF8 | ConvertFrom-Json
        if (-not [bool]$payload.complete) { return $false }
        if ([int]$payload.horizon -ne $Horizon) { return $false }
        if ([int]$payload.result_rows -le 0) { return $false }
        # The metadata is written last by _save_checkpoint.  Require one stable
        # observation of all three files so a watcher never races file commits.
        $first = @($family, $yearly, $meta) | ForEach-Object {
            $item = Get-Item -LiteralPath $_
            "$($_):$($item.Length):$($item.LastWriteTimeUtc.Ticks)"
        }
        Start-Sleep -Milliseconds 250
        $second = @($family, $yearly, $meta) | ForEach-Object {
            $item = Get-Item -LiteralPath $_
            "$($_):$($item.Length):$($item.LastWriteTimeUtc.Ticks)"
        }
        return (@($first) -join "|") -eq (@($second) -join "|")
    } catch {
        return $false
    }
}

function Wait-ForBoundaries {
    while ($true) {
        $states = @{}
        foreach ($horizon in @($BoundaryHorizons | Sort-Object -Unique)) {
            $states[[string]$horizon] = Test-HorizonComplete -Horizon $horizon
        }
        $complete = @($states.GetEnumerator() | Where-Object Value | ForEach-Object Key)
        Write-WatcherEvent "boundary_poll" @{
            complete_horizons = $complete
            pending_horizons = @($states.GetEnumerator() | Where-Object { -not $_.Value } | ForEach-Object Key)
        }
        if ($complete.Count -eq $states.Count) { return }
        Start-Sleep -Seconds $PollSeconds
    }
}

function Assert-ExpectedParent {
    $row = @(Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $ParentPid)) | Select-Object -First 1
    if ($null -eq $row) { throw "HORIZON_WATCHER_PARENT_NOT_RUNNING:$ParentPid" }
    $commandLine = [string]$row.CommandLine
    if ($commandLine -notmatch "dynamic_qbd_full_space_fold_clock_validation") {
        throw "HORIZON_WATCHER_UNEXPECTED_PARENT_COMMANDLINE:$commandLine"
    }
    return $row
}

function Stop-ValidatedProcessTree {
    $root = @(Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f $ParentPid)) | Select-Object -First 1
    if ($null -eq $root) {
        Write-WatcherEvent "parent_already_stopped"
        return
    }
    $validated = Assert-ExpectedParent
    $ids = @(Get-ProcessTreeIds -Root $ParentPid)
    Write-WatcherEvent "stopping_parent_tree" @{ process_ids = $ids; command_line = [string]$validated.CommandLine }
    foreach ($id in @($ids | Sort-Object -Descending)) {
        try { Stop-Process -Id ([int]$id) -Force -ErrorAction Stop } catch { }
    }
    while (@(Get-ProcessSnapshot | Where-Object { $ids -contains [int]$_.ProcessId }).Count -gt 0) {
        Start-Sleep -Milliseconds 250
    }
}

$resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
$resolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot -ErrorAction SilentlyContinue)
if ($null -eq $resolvedOutputRoot) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    $resolvedOutputRoot = (Resolve-Path -LiteralPath $OutputRoot).Path
} else {
    $resolvedOutputRoot = $resolvedOutputRoot.Path
}
if (@($BoundaryHorizons | Where-Object { $_ -lt 1 -or $_ -gt 30 }).Count -gt 0) {
    throw "HORIZON_WATCHER_BOUNDARY_OUT_OF_RANGE"
}
if ($ResumeArguments.Count -eq 0) { throw "HORIZON_WATCHER_RESUME_ARGUMENTS_EMPTY" }

Write-WatcherEvent "watcher_started" @{ working_directory = $resolvedWorkingDirectory; output_root = $resolvedOutputRoot }
Wait-ForBoundaries
Write-WatcherEvent "boundary_reached"

if (-not $StopParentAtBoundary) {
    Write-WatcherEvent "boundary_reached_no_switch" @{ reason = "STOP_PARENT_AT_BOUNDARY_SWITCH_REQUIRED" }
    exit 2
}

Stop-ValidatedProcessTree
$stdout = Join-Path $resolvedOutputRoot "horizon-watcher-resume.stdout.log"
$stderr = Join-Path $resolvedOutputRoot "horizon-watcher-resume.stderr.log"
$child = Start-Process -FilePath $PythonExecutable -ArgumentList $ResumeArguments `
    -WorkingDirectory $resolvedWorkingDirectory -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -PassThru
Write-WatcherEvent "resume_started" @{
    resume_pid = [int]$child.Id
    python_executable = $PythonExecutable
    resume_arguments = @($ResumeArguments)
    stdout = $stdout
    stderr = $stderr
}

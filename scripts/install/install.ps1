# NeoCore: one command on Windows (PowerShell 5.1 or 7).
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 setup
#   powershell -ExecutionPolicy Bypass -File install.ps1 connect --url URL --key-file FILE
#
# Makes (or reuses) a private Python in %USERPROFILE%\.neocore\venv, installs NeoCore into it
# and runs `neocore` with the arguments given. Running it again upgrades and changes nothing
# twice. NEOCORE_PACKAGE picks what to install (a wheel file, for instance).
$ErrorActionPreference = 'Stop'

$Package = if ($env:NEOCORE_PACKAGE) { $env:NEOCORE_PACKAGE } else { 'neocore-memory[embed]' }
$Venv = if ($env:NEOCORE_VENV) { $env:NEOCORE_VENV } else { Join-Path $HOME '.neocore\venv' }
$Python = Join-Path $Venv 'Scripts\python.exe'

function Fail($message) {
    [Console]::Error.WriteLine("neocore install: $message")
    exit 1
}

if ($args.Count -eq 0) { Fail 'say what to set up: setup, connect or server init (see the top of this file)' }
$uv = Get-Command uv -ErrorAction SilentlyContinue

if (-not (Test-Path $Python)) {
    New-Item -ItemType Directory -Force (Split-Path $Venv) | Out-Null
    if ($uv) {
        & uv venv -q --python '>=3.10' $Venv
        if ($LASTEXITCODE -ne 0) { Fail "uv could not make $Venv" }
    } else {
        $found = $null
        foreach ($version in '-3.13', '-3.12', '-3.11', '-3.10') {
            if (Get-Command py -ErrorAction SilentlyContinue) {
                & py $version -c 'import sys' 2>$null
                if ($LASTEXITCODE -eq 0) { $found = @('py', $version); break }
            }
        }
        if (-not $found) { Fail 'NeoCore needs Python 3.10 or newer (or uv): install one and run this again' }
        & $found[0] $found[1] -m venv $Venv
        if ($LASTEXITCODE -ne 0) { Fail "could not make $Venv" }
    }
}

Write-Host "Installing $Package into $Venv ..."
if ($uv) {
    $env:VIRTUAL_ENV = $Venv
    & uv pip install -q --upgrade $Package
} else {
    & $Python -m pip install -q --upgrade $Package
}
if ($LASTEXITCODE -ne 0) { Fail 'install failed' }

& $Python -m neocore @args
exit $LASTEXITCODE

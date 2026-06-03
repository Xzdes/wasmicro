param(
    [string]$OutDir = "demo/pkg",
    [string]$OutName = "wasmicro",
    [int]$WasmBudgetKb = 250
)

$ErrorActionPreference = "Stop"

function Format-Bytes([long]$Bytes) {
    $culture = [Globalization.CultureInfo]::InvariantCulture
    if ($Bytes -ge 1MB) {
        return [string]::Format($culture, "{0:N1} MB", ($Bytes / 1MB))
    }
    return [string]::Format($culture, "{0:N1} KB", ($Bytes / 1KB))
}

function Get-FileSize([string]$Path) {
    if (Test-Path $Path) {
        return (Get-Item $Path).Length
    }
    return 0
}

function Invoke-Checked([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

function Remove-GeneratedWasmPackFiles([string]$Dir, [string]$Name) {
    if (!(Test-Path $Dir)) {
        return
    }
    $generated = @(
        "${Name}.js",
        "${Name}.d.ts",
        "${Name}_bg.js",
        "${Name}_bg.wasm",
        "${Name}_bg.wasm.d.ts"
    )
    foreach ($file in $generated) {
        $path = Join-Path $Dir $file
        if (Test-Path $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
}

Write-Host "[1/5] Building WASM package ..."
Remove-GeneratedWasmPackFiles $OutDir $OutName
Invoke-Checked "wasm-pack" @(
    "build",
    "--release",
    "--target", "web",
    "--no-opt",
    "--out-dir", $OutDir,
    "--out-name", $OutName,
    "--features", "wasm"
)

$wasmPath = Join-Path $OutDir "${OutName}_bg.wasm"
if (Get-Command wasm-opt -ErrorAction SilentlyContinue) {
    Write-Host "[2/5] Optimizing WASM with wasm-opt -Oz ..."
    Invoke-Checked "wasm-opt" @(
        "--enable-bulk-memory",
        "--enable-nontrapping-float-to-int",
        "--enable-simd",
        "-Oz",
        $wasmPath,
        "-o",
        $wasmPath
    )
} else {
    Write-Host "[2/5] wasm-opt not found; reporting unoptimized WASM size."
}

Write-Host "[3/5] Normalizing npm package metadata ..."
Push-Location $OutDir
try {
    Invoke-Checked "npm" @("pkg", "fix")
} finally {
    Pop-Location
}

Write-Host "[4/5] Measuring package files ..."
$wasmBytes = Get-FileSize $wasmPath
$jsBytes = (Get-ChildItem $OutDir -Filter "*.js" | Measure-Object -Property Length -Sum).Sum
$dtsBytes = (Get-ChildItem $OutDir -Filter "*.d.ts" | Measure-Object -Property Length -Sum).Sum

Write-Host "[5/5] Measuring npm dry-run tarball ..."
Push-Location $OutDir
try {
    $packJson = npm pack --dry-run --json | ConvertFrom-Json
    $tarballBytes = [long]$packJson[0].size
    $unpackedBytes = [long]$packJson[0].unpackedSize
} finally {
    Pop-Location
}

$budgetBytes = $WasmBudgetKb * 1KB
$overBudget = $wasmBytes -gt $budgetBytes

Write-Host ""
Write-Host "Size report"
Write-Host "-----------"
Write-Host ("WASM          {0}" -f (Format-Bytes $wasmBytes))
Write-Host ("JS glue       {0}" -f (Format-Bytes $jsBytes))
Write-Host ("Type defs     {0}" -f (Format-Bytes $dtsBytes))
Write-Host ("npm tarball   {0}" -f (Format-Bytes $tarballBytes))
Write-Host ("npm unpacked  {0}" -f (Format-Bytes $unpackedBytes))
Write-Host ("WASM budget   {0}" -f (Format-Bytes $budgetBytes))

if ($overBudget) {
    Write-Host "Status        over budget"
    exit 1
}

Write-Host "Status        within budget"

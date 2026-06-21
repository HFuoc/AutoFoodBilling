param(
    [int]$Epochs = 20,
    [int]$BatchSize = 16,
    [int]$ImageSize = 160,
    [int]$MaxSupplementPerClass = 250,
    [string]$Annotations = "data\annotations\tray_cells_fixed_template.json",
    [string]$PrimaryCrops = "data\processed\tray_cell_food_classes_fixed",
    [string]$TrainingData = "data\processed\tray_cell_food_classes_supplemented",
    [string]$ModelOutput = "ml\models\cnn\cell_best.cnn",
    [switch]$SkipDataPrep,
    [switch]$SkipSmokeTest,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DemoImage = Join-Path $ProjectRoot "frontend\public\demo-tray.png"
$ModelPath = Join-Path $ProjectRoot $ModelOutput
$ModelMeta = [System.IO.Path]::ChangeExtension($ModelPath, ".json")

function Invoke-Step {
    param(
        [string]$Title,
        [string[]]$Command
    )

    Write-Host ""
    Write-Host "==> $Title"
    Write-Host ($Command -join " ")
    if ($DryRun) {
        return
    }

    $exe = $Command[0]
    $args = if ($Command.Count -gt 1) { $Command[1..($Command.Count - 1)] } else { @() }
    & $exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title"
    }
}

Set-Location $ProjectRoot
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path $Python)) {
    throw "Khong tim thay .venv. Hay chay: python -m venv .venv; .\.venv\Scripts\pip.exe install -r backend\requirements.txt"
}

if (-not $SkipDataPrep) {
    Invoke-Step "Extract labeled tray-cell crops" @(
        $Python,
        "ml\training\extract_tray_cell_crops.py",
        "--annotations",
        $Annotations,
        "--output",
        $PrimaryCrops,
        "--overwrite"
    )

    Invoke-Step "Build supplemented CNN dataset" @(
        $Python,
        "ml\training\prepare_supplemented_cnn_dataset.py",
        "--primary",
        $PrimaryCrops,
        "--supplement",
        "data\raw\food_classes",
        "--output",
        $TrainingData,
        "--max-supplement-per-class",
        [string]$MaxSupplementPerClass,
        "--overwrite"
    )
}

Invoke-Step "Train cell CNN checkpoint" @(
    $Python,
    "-u",
    "ml\training\train_cnn.py",
    "--data-root",
    $TrainingData,
    "--classes",
    (Join-Path $TrainingData "classes.json"),
    "--output",
    $ModelOutput,
    "--epochs",
    [string]$Epochs,
    "--batch-size",
    [string]$BatchSize,
    "--image-size",
    [string]$ImageSize
)

if (-not $DryRun -and -not (Test-Path $ModelPath)) {
    throw "Train finished but model was not created: $ModelPath"
}
if (-not $DryRun -and -not (Test-Path $ModelMeta)) {
    throw "Train finished but model metadata was not created: $ModelMeta"
}

if (-not $SkipSmokeTest -and (Test-Path $DemoImage)) {
    Invoke-Step "Smoke test demo image" @(
        $Python,
        "ml\inference\run_cell_pipeline.py",
        $DemoImage,
        "--cnn-model",
        $ModelOutput,
        "--crop-mode",
        "template"
    )
}

Write-Host ""
Write-Host "==> Start backend + frontend"
if ($DryRun) {
    Write-Host ((Join-Path $ProjectRoot "run_project.ps1") + " -NoBrowser:" + $NoBrowser)
} else {
    & (Join-Path $ProjectRoot "run_project.ps1") -NoBrowser:$NoBrowser
}

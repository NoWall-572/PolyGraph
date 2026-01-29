# 组织数据集到 ASTRA_Release 的 PowerShell 脚本
# 创建合理的目录结构并移动数据集

$baseDir = "E:\Koi-data\Pandora's box\Multi-Agent System Fault Diagnosis\autodl-agents-main\Agents_Failure_Attribution-main"
$releaseDir = Join-Path $baseDir "ASTRA_Release"

Write-Host "正在创建目录结构..." -ForegroundColor Green

# 创建数据目录结构
$dataDirs = @(
    "data\raw\whowhen",
    "data\raw\tracertraj",
    "data\processed\whowhen",
    "data\processed\tracertraj"
)

foreach ($dir in $dataDirs) {
    $fullPath = Join-Path $releaseDir $dir
    New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
    Write-Host "  ✓ 创建目录: $dir" -ForegroundColor Cyan
}

Write-Host "`n正在移动数据集..." -ForegroundColor Green

# 1. 移动 Who&When 原始数据
$source = Join-Path $baseDir "Who&When"
$target = Join-Path $releaseDir "data\raw\whowhen"
if (Test-Path $source) {
    Write-Host "  移动 Who&When 原始数据..." -ForegroundColor Yellow
    Copy-Item -Path $source -Destination $target -Recurse -Force
    Write-Host "  ✓ Who&When 原始数据已复制到 data/raw/whowhen/" -ForegroundColor Cyan
} else {
    Write-Host "  ⚠ Who&When 目录不存在: $source" -ForegroundColor Red
}

# 2. 移动 graphs_whowhen_ollama (转换后的 benchmark 图数据)
$source = Join-Path $baseDir "processed_graphs\graphs_whowhen_ollama"
$target = Join-Path $releaseDir "data\processed\whowhen\graphs_whowhen_ollama"
if (Test-Path $source) {
    Write-Host "  移动 graphs_whowhen_ollama..." -ForegroundColor Yellow
    Copy-Item -Path $source -Destination $target -Recurse -Force
    Write-Host "  ✓ graphs_whowhen_ollama 已复制到 data/processed/whowhen/" -ForegroundColor Cyan
} else {
    Write-Host "  ⚠ graphs_whowhen_ollama 目录不存在: $source" -ForegroundColor Red
}

# 3. 移动 AgenTracer 原始数据
$source = Join-Path $baseDir "dataset_raw\AgenTracer"
$target = Join-Path $releaseDir "data\raw\tracertraj"
if (Test-Path $source) {
    Write-Host "  移动 AgenTracer 原始数据..." -ForegroundColor Yellow
    Copy-Item -Path $source -Destination $target -Recurse -Force
    Write-Host "  ✓ AgenTracer 原始数据已复制到 data/raw/tracertraj/" -ForegroundColor Cyan
} else {
    Write-Host "  ⚠ AgenTracer 目录不存在: $source" -ForegroundColor Red
}

# 4. 移动 graphs_tracertraj (处理后的图数据)
$source = Join-Path $baseDir "processed_graphs\graphs_tracertraj"
$target = Join-Path $releaseDir "data\processed\tracertraj\graphs_tracertraj"
if (Test-Path $source) {
    Write-Host "  移动 graphs_tracertraj..." -ForegroundColor Yellow
    Copy-Item -Path $source -Destination $target -Recurse -Force
    Write-Host "  ✓ graphs_tracertraj 已复制到 data/processed/tracertraj/" -ForegroundColor Cyan
} else {
    Write-Host "  ⚠ graphs_tracertraj 目录不存在: $source" -ForegroundColor Red
}

Write-Host "`n✅ 数据集组织完成！" -ForegroundColor Green
Write-Host "`n目录结构:" -ForegroundColor Cyan
Write-Host "  ASTRA_Release/" -ForegroundColor White
Write-Host "    data/" -ForegroundColor White
Write-Host "      raw/" -ForegroundColor White
Write-Host "        whowhen/          (Who&When 原始数据)" -ForegroundColor Gray
Write-Host "        tracertraj/       (AgenTracer 原始数据)" -ForegroundColor Gray
Write-Host "      processed/" -ForegroundColor White
Write-Host "        whowhen/" -ForegroundColor White
Write-Host "          graphs_whowhen_ollama/  (转换后的 benchmark 图数据)" -ForegroundColor Gray
Write-Host "        tracertraj/" -ForegroundColor White
Write-Host "          graphs_tracertraj/       (处理后的图数据)" -ForegroundColor Gray


<#
.SYNOPSIS
XingClaw Windows 本地调试启动脚本
.DESCRIPTION
使用方式:
  .\dev.ps1                       # 默认 IM webhook 模式
  .\dev.ps1 -Mode cli             # CLI 交互模式
  .\dev.ps1 -Mode im -Transport longconn  # 飞书长连接模式
#>
param(
    [string]$Mode       = "im",
    [string]$Transport  = "webhook",
    [string]$ListenHost = "127.0.0.1",
    [int]   $Port       = 8787,
    [string]$Workspace  = ".",
    [string]$LogLevel   = "debug"
)

$ErrorActionPreference = "Stop"

# 加载 .env.ps1（如果存在）
$envFile = Join-Path $PSScriptRoot ".env.ps1"
if (Test-Path $envFile) {
    Write-Host "[dev] Loading $envFile ..." -ForegroundColor Cyan
    . $envFile
}

# 确保在项目根目录
Set-Location $PSScriptRoot

# 确保已安装（开发模式）
$installed = pip show xingclaw 2>$null
if (-not $installed) {
    Write-Host "[dev] Installing xingclaw in editable mode ..." -ForegroundColor Yellow
    pip install -e ".[dev]"
}

$provider = if ($env:XINGCLAW_PROVIDER) { $env:XINGCLAW_PROVIDER } else { "anthropic" }
$modelId  = if ($env:XINGCLAW_MODEL_ID) { $env:XINGCLAW_MODEL_ID } else { "claude-sonnet-4-5" }

if ($Mode -eq "im") {
    $appId     = $env:FEISHU_APP_ID
    $appSecret = $env:FEISHU_APP_SECRET
    $verifyTk  = $env:FEISHU_VERIFY_TOKEN

    if (-not $appId -or -not $appSecret) {
        Write-Host "[dev] ERROR: FEISHU_APP_ID and FEISHU_APP_SECRET must be set." -ForegroundColor Red
        Write-Host "[dev] Create .env.ps1 from .env.ps1.example and fill in values." -ForegroundColor Red
        exit 1
    }

    $pyArgs = @(
        "-m", "im",
        "--platform", "feishu",
        "--transport", $Transport,
        "--workspace", $Workspace,
        "--host", $ListenHost,
        "--port", $Port,
        "--provider", $provider,
        "--model-id", $modelId,
        "--feishu-app-id", $appId,
        "--feishu-app-secret", $appSecret,
        "--log-level", $LogLevel
    )
    if ($verifyTk) {
        $pyArgs += @("--feishu-verify-token", $verifyTk)
    }

    Write-Host "[dev] Starting IM service ($Transport) on ${ListenHost}:${Port} ..." -ForegroundColor Green
    Write-Host "[dev] Provider: $provider | Model: $modelId" -ForegroundColor Green
    python @pyArgs

} else {
    $pyArgs = @(
        "-m", "coding_agent",
        "--mode", "interactive",
        "--workspace", $Workspace,
        "--provider", $provider,
        "--model-id", $modelId
    )

    Write-Host "[dev] Starting CLI interactive mode ..." -ForegroundColor Green
    Write-Host "[dev] Provider: $provider | Model: $modelId" -ForegroundColor Green
    python @pyArgs
}

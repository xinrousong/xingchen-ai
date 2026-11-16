#Requires -RunAsAdministrator
<#
  星辰 · GPU 恢复脚本 v2（Code 43 / CM_PROB_FAILED_POST_START）

  v2 相比 v1 增加：禁用 GameViewer 虚拟显示适配器。
  它虽然进程没跑，但驱动仍在加载，是 GeForce 卡 Code 43 的常见冲突源。

  诊断依据（2026-11-16 23:20）：
    - nvlddmkm.sys 在 DriverStore\nvtfi.inf_amd64_37f842ed1c0b9b30 中（97.5MB，11-09）
    - oem109.inf DriverVer = 32.0.16.2124（621.24），已生效
    - Services\nvlddmkm ImagePath 指向该文件，且文件存在
    - 但设备 Device Status = 0x01806400（含 DN_HAS_PROBLEM），启动失败
    - NVDisplay 日志：NvAPI_Initialize -6（找不到设备）
  结论：文件齐全，是初始化阶段失败，先排除虚拟显示器冲突。

  用法：
    右键本文件 → 使用 PowerShell 运行
    或管理员 PowerShell：
      powershell -ExecutionPolicy Bypass -File D:\dsh-demo\xingchen\scripts\fix_gpu.ps1

  屏幕会黑 3~5 秒，属正常。全部可逆。
#>

$ErrorActionPreference = 'Continue'
$devId = 'PCI\VEN_10DE&DEV_2D19&SUBSYS_60501D05&REV_A1\04333D4F6D2DB04800'

function Step($n, $t) { Write-Host "`n[$n] $t" -ForegroundColor Cyan }

Step 1 "停止 GameViewer 服务并禁止自启"
$svc = Get-Service -Name 'GameViewerService' -ErrorAction SilentlyContinue
if ($svc) {
    try {
        Stop-Service -Name 'GameViewerService' -Force -ErrorAction Stop
        Set-Service  -Name 'GameViewerService' -StartupType Manual -ErrorAction Stop
        Write-Host "    GameViewerService：已停止，启动类型改为 Manual" -ForegroundColor Green
    } catch {
        Write-Host "    停止服务失败：$($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "    未找到 GameViewerService"
}
Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'GameViewer*' } |
    ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }

Step 2 "禁用 GameViewer 虚拟显示适配器（v2 新增）"
$vgpu = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
        Where-Object { $_.FriendlyName -like '*GameViewer*' }
if ($vgpu) {
    try {
        Disable-PnpDevice -InstanceId $vgpu.InstanceId -Confirm:$false -ErrorAction Stop
        Write-Host "    已禁用：$($vgpu.FriendlyName)" -ForegroundColor Green
        Write-Host "    恢复远程串流：Enable-PnpDevice -InstanceId '$($vgpu.InstanceId)'" -ForegroundColor DarkGray
    } catch {
        Write-Host "    禁用失败：$($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "    没有 GameViewer 虚拟显示设备"
}

Step 3 "重置 NVIDIA 显卡设备（屏幕会黑 3~5 秒）"
$nv = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
      Where-Object { $_.FriendlyName -like '*NVIDIA*' }
if (-not $nv) { Write-Host "    找不到 NVIDIA 显示设备，中止" -ForegroundColor Red; exit 1 }
Write-Host "    当前状态：$($nv.Status) / $($nv.Problem)"

try {
    Disable-PnpDevice -InstanceId $devId -Confirm:$false -ErrorAction Stop
    Start-Sleep -Seconds 3
    Enable-PnpDevice  -InstanceId $devId -Confirm:$false -ErrorAction Stop
    Write-Host "    已重置，等待驱动初始化..." -ForegroundColor Green
} catch {
    Write-Host "    重置失败：$($_.Exception.Message)" -ForegroundColor Red
}
Start-Sleep -Seconds 8

Step 4 "验证"
$nv2 = Get-PnpDevice -Class Display -ErrorAction SilentlyContinue |
       Where-Object { $_.FriendlyName -like '*NVIDIA*' }
Write-Host "    设备状态：$($nv2.Status)  问题码：$($nv2.Problem)"
$svc2 = Get-Service nvlddmkm -ErrorAction SilentlyContinue
Write-Host "    nvlddmkm 服务：$($svc2.Status)"
Write-Host ""
Write-Host "    nvidia-smi：" -NoNewline
$smi = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host $smi -ForegroundColor Green
    Write-Host "`n[OK] 显卡已恢复。回对话里说一声，我立刻验证 torch 并跑基准测试。" -ForegroundColor Green
} else {
    Write-Host "仍失败" -ForegroundColor Red
    Write-Host "    输出：$smi"
    Write-Host @"

    [!] 重置设备没解决。按代价从低到高：
       1. 重启电脑（装完驱动后必须重启才真正初始化设备）。
       2. 仍 Code 43 → 本机 HypervisorPresent=True，可能是 Hyper-V 抢占。
          管理员 CMD：
              bcdedit /set hypervisorlaunchtype off
          重启。代价：WSL2 / Docker Desktop / Windows 沙盒不可用，改回 auto 可恢复。
       3. 仍无效 → DDU 安全模式清理后重装。本地已有完整安装包：
          C:\ProgramData\NVIDIA Corporation\NVIDIA App\UpdateFramework\ota-artifacts\grd\
          ea135b338ec3d58d1d4e6bda7e506dc0\621.24-notebook-win10-win11-64bit-international-dch-whql-g.exe
       4. 以上全无效 → 考虑硬件排查（该卡曾正常工作，但仍需排除）。
"@ -ForegroundColor Yellow
}

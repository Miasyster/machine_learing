# Airflow ML项目部署脚本 (PowerShell)
# 用于Windows环境下的Docker部署

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "stop", "restart", "build", "logs", "status", "clean", "init")]
    [string]$Action = "start",
    
    [Parameter(Mandatory=$false)]
    [string]$Service = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$Build = $false,
    
    [Parameter(Mandatory=$false)]
    [switch]$Detach = $true,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force = $false
)

# 设置错误处理
$ErrorActionPreference = "Stop"

# 颜色输出函数
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✓ $Message" "Green"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "✗ $Message" "Red"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠ $Message" "Yellow"
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput "ℹ $Message" "Cyan"
}

# 检查Docker是否安装
function Test-Docker {
    try {
        docker --version | Out-Null
        docker-compose --version | Out-Null
        return $true
    }
    catch {
        Write-Error "Docker或Docker Compose未安装或未启动"
        Write-Info "请安装Docker Desktop并确保其正在运行"
        return $false
    }
}

# 检查环境文件
function Test-Environment {
    if (-not (Test-Path ".env")) {
        if (Test-Path ".env.example") {
            Write-Warning "未找到.env文件，正在从.env.example创建..."
            Copy-Item ".env.example" ".env"
            Write-Info "请编辑.env文件并设置正确的环境变量"
        }
        else {
            Write-Error "未找到.env.example文件"
            return $false
        }
    }
    return $true
}

# 创建必要的目录
function Initialize-Directories {
    Write-Info "创建必要的目录..."
    
    $directories = @(
        "data/raw",
        "data/processed", 
        "data/features",
        "models/trained",
        "models/staging",
        "models/production",
        "logs/ml/training",
        "logs/ml/inference",
        "logs/ml/monitoring",
        "notebooks",
        "plugins/ml"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "创建目录: $dir"
        }
    }
}

# 设置权限（Windows下的等效操作）
function Set-Permissions {
    Write-Info "设置目录权限..."
    
    # 在Windows下，我们主要确保目录存在且可写
    $directories = @("data", "models", "logs", "notebooks")
    
    foreach ($dir in $directories) {
        if (Test-Path $dir) {
            # 确保当前用户有完全控制权限
            try {
                $acl = Get-Acl $dir
                $accessRule = New-Object System.Security.AccessControl.FileSystemAccessRule(
                    [System.Security.Principal.WindowsIdentity]::GetCurrent().Name,
                    "FullControl",
                    "ContainerInherit,ObjectInherit",
                    "None",
                    "Allow"
                )
                $acl.SetAccessRule($accessRule)
                Set-Acl -Path $dir -AclObject $acl
                Write-Success "设置权限: $dir"
            }
            catch {
                Write-Warning "无法设置权限: $dir"
            }
        }
    }
}

# 构建镜像
function Build-Images {
    Write-Info "构建Docker镜像..."
    
    try {
        if ($Force) {
            docker-compose build --no-cache
        }
        else {
            docker-compose build
        }
        Write-Success "镜像构建完成"
    }
    catch {
        Write-Error "镜像构建失败: $_"
        exit 1
    }
}

# 启动服务
function Start-Services {
    Write-Info "启动Airflow服务..."
    
    try {
        if ($Build) {
            Build-Images
        }
        
        if ($Service) {
            if ($Detach) {
                docker-compose up -d $Service
            }
            else {
                docker-compose up $Service
            }
        }
        else {
            if ($Detach) {
                docker-compose up -d
            }
            else {
                docker-compose up
            }
        }
        
        Write-Success "服务启动完成"
        Write-Info "Airflow Web界面: http://localhost:8080"
        Write-Info "默认用户名/密码: admin/admin123"
        Write-Info "Flower监控界面: http://localhost:5555 (如果启用)"
        Write-Info "Jupyter Notebook: http://localhost:8888 (如果启用)"
        Write-Info "MLflow界面: http://localhost:5000 (如果启用)"
    }
    catch {
        Write-Error "服务启动失败: $_"
        exit 1
    }
}

# 停止服务
function Stop-Services {
    Write-Info "停止Airflow服务..."
    
    try {
        if ($Service) {
            docker-compose stop $Service
        }
        else {
            docker-compose down
        }
        Write-Success "服务停止完成"
    }
    catch {
        Write-Error "服务停止失败: $_"
        exit 1
    }
}

# 重启服务
function Restart-Services {
    Write-Info "重启Airflow服务..."
    Stop-Services
    Start-Services
}

# 查看日志
function Show-Logs {
    try {
        if ($Service) {
            docker-compose logs -f $Service
        }
        else {
            docker-compose logs -f
        }
    }
    catch {
        Write-Error "查看日志失败: $_"
        exit 1
    }
}

# 查看状态
function Show-Status {
    Write-Info "服务状态:"
    
    try {
        docker-compose ps
        
        Write-Info "`n容器资源使用情况:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
        
        Write-Info "`n磁盘使用情况:"
        docker system df
    }
    catch {
        Write-Error "查看状态失败: $_"
        exit 1
    }
}

# 清理资源
function Clean-Resources {
    Write-Warning "这将删除所有容器、镜像和数据卷！"
    
    if (-not $Force) {
        $confirmation = Read-Host "确定要继续吗？(y/N)"
        if ($confirmation -ne "y" -and $confirmation -ne "Y") {
            Write-Info "操作已取消"
            return
        }
    }
    
    Write-Info "清理Docker资源..."
    
    try {
        # 停止并删除容器
        docker-compose down -v --remove-orphans
        
        # 删除镜像
        $images = docker images --filter "reference=*airflow*" -q
        if ($images) {
            docker rmi $images -f
        }
        
        # 清理未使用的资源
        docker system prune -f
        
        Write-Success "清理完成"
    }
    catch {
        Write-Error "清理失败: $_"
        exit 1
    }
}

# 初始化项目
function Initialize-Project {
    Write-Info "初始化ML项目..."
    
    # 检查环境
    if (-not (Test-Environment)) {
        exit 1
    }
    
    # 创建目录
    Initialize-Directories
    
    # 设置权限
    Set-Permissions
    
    # 构建镜像
    Build-Images
    
    # 启动初始化服务
    Write-Info "启动初始化服务..."
    docker-compose up -d postgres redis ml-postgres ml-redis
    
    # 等待数据库就绪
    Write-Info "等待数据库就绪..."
    Start-Sleep -Seconds 30
    
    # 初始化Airflow数据库
    Write-Info "初始化Airflow数据库..."
    docker-compose run --rm airflow-init
    
    Write-Success "项目初始化完成"
    Write-Info "现在可以运行: .\deploy.ps1 start"
}

# 主函数
function Main {
    Write-Info "Airflow ML项目部署脚本"
    Write-Info "操作: $Action"
    
    # 检查Docker
    if (-not (Test-Docker)) {
        exit 1
    }
    
    # 切换到脚本目录
    Set-Location $PSScriptRoot
    
    # 执行操作
    switch ($Action.ToLower()) {
        "start" {
            if (-not (Test-Environment)) { exit 1 }
            Start-Services
        }
        "stop" {
            Stop-Services
        }
        "restart" {
            Restart-Services
        }
        "build" {
            Build-Images
        }
        "logs" {
            Show-Logs
        }
        "status" {
            Show-Status
        }
        "clean" {
            Clean-Resources
        }
        "init" {
            Initialize-Project
        }
        default {
            Write-Error "未知操作: $Action"
            Write-Info "可用操作: start, stop, restart, build, logs, status, clean, init"
            exit 1
        }
    }
}

# 脚本使用说明
function Show-Help {
    Write-Host @"
Airflow ML项目部署脚本

用法:
    .\deploy.ps1 [操作] [选项]

操作:
    init        初始化项目（首次运行）
    start       启动服务（默认）
    stop        停止服务
    restart     重启服务
    build       构建镜像
    logs        查看日志
    status      查看状态
    clean       清理资源

选项:
    -Service    指定服务名称
    -Build      启动时重新构建镜像
    -Detach     后台运行（默认）
    -Force      强制执行（跳过确认）

示例:
    .\deploy.ps1 init                    # 初始化项目
    .\deploy.ps1 start                   # 启动所有服务
    .\deploy.ps1 start -Service webserver # 只启动web服务
    .\deploy.ps1 logs -Service scheduler  # 查看调度器日志
    .\deploy.ps1 build -Force            # 强制重新构建
    .\deploy.ps1 clean -Force            # 强制清理所有资源

"@ -ForegroundColor White
}

# 检查是否需要显示帮助
if ($args -contains "-h" -or $args -contains "--help" -or $args -contains "help") {
    Show-Help
    exit 0
}

# 运行主函数
try {
    Main
}
catch {
    Write-Error "脚本执行失败: $_"
    exit 1
}
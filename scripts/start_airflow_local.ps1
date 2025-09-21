# Airflow 本地启动脚本
param(
    [string]$Action = "start"
)

# 设置错误处理
$ErrorActionPreference = "Stop"

# 颜色输出函数
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Info($message) {
    Write-ColorOutput Cyan "ℹ️  $message"
}

function Write-Success($message) {
    Write-ColorOutput Green "✅ $message"
}

function Write-Warning($message) {
    Write-ColorOutput Yellow "⚠️  $message"
}

function Write-Error($message) {
    Write-ColorOutput Red "❌ $message"
}

# 设置环境变量
$env:AIRFLOW_HOME = "$PWD\airflow_home"
$env:AIRFLOW__CORE__DAGS_FOLDER = "$PWD\dags"
$env:AIRFLOW__CORE__LOAD_EXAMPLES = "False"
$env:AIRFLOW__WEBSERVER__EXPOSE_CONFIG = "True"
$env:AIRFLOW__CORE__EXECUTOR = "SequentialExecutor"
$env:AIRFLOW__DATABASE__SQL_ALCHEMY_CONN = "sqlite:///$($env:AIRFLOW_HOME)/airflow.db"

Write-Info "Airflow 本地部署脚本"
Write-Info "AIRFLOW_HOME: $env:AIRFLOW_HOME"
Write-Info "DAGS_FOLDER: $env:AIRFLOW__CORE__DAGS_FOLDER"

# 创建必要的目录
if (!(Test-Path $env:AIRFLOW_HOME)) {
    Write-Info "创建 Airflow 主目录..."
    New-Item -ItemType Directory -Path $env:AIRFLOW_HOME -Force | Out-Null
}

if (!(Test-Path "$PWD\logs")) {
    Write-Info "创建日志目录..."
    New-Item -ItemType Directory -Path "$PWD\logs" -Force | Out-Null
}

switch ($Action.ToLower()) {
    "install" {
        Write-Info "安装 Airflow..."
        
        # 检查是否已安装
        try {
            $airflowVersion = python -c "import airflow; print(airflow.__version__)" 2>$null
            if ($airflowVersion) {
                Write-Success "Airflow 已安装，版本: $airflowVersion"
                return
            }
        } catch {
            # 继续安装
        }
        
        Write-Info "安装 Apache Airflow..."
        pip install apache-airflow==2.7.0 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.7.0/constraints-3.10.txt"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Airflow 安装成功！"
        } else {
            Write-Error "Airflow 安装失败！"
            exit 1
        }
    }
    
    "init" {
        Write-Info "初始化 Airflow 数据库..."
        airflow db init
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "数据库初始化成功！"
            
            Write-Info "创建管理员用户..."
            airflow users create `
                --username admin `
                --firstname Admin `
                --lastname User `
                --role Admin `
                --email admin@example.com `
                --password admin
                
            if ($LASTEXITCODE -eq 0) {
                Write-Success "管理员用户创建成功！"
                Write-Info "用户名: admin"
                Write-Info "密码: admin"
            }
        } else {
            Write-Error "数据库初始化失败！"
            exit 1
        }
    }
    
    "start" {
        Write-Info "启动 Airflow..."
        
        # 检查是否已初始化
        if (!(Test-Path "$env:AIRFLOW_HOME\airflow.db")) {
            Write-Warning "数据库未初始化，正在初始化..."
            & $PSCommandPath -Action init
        }
        
        Write-Info "启动 Airflow Web 服务器..."
        Write-Info "Web 界面将在 http://localhost:8080 启动"
        Write-Info "用户名: admin, 密码: admin"
        Write-Info "按 Ctrl+C 停止服务"
        
        # 启动调度器（后台）
        Start-Process -FilePath "airflow" -ArgumentList "scheduler" -WindowStyle Hidden
        
        # 启动 Web 服务器（前台）
        airflow webserver --port 8080
    }
    
    "scheduler" {
        Write-Info "启动 Airflow 调度器..."
        airflow scheduler
    }
    
    "webserver" {
        Write-Info "启动 Airflow Web 服务器..."
        airflow webserver --port 8080
    }
    
    "stop" {
        Write-Info "停止 Airflow 服务..."
        Get-Process | Where-Object {$_.ProcessName -like "*airflow*"} | Stop-Process -Force
        Write-Success "Airflow 服务已停止"
    }
    
    "status" {
        Write-Info "检查 Airflow 服务状态..."
        $processes = Get-Process | Where-Object {$_.ProcessName -like "*airflow*"}
        if ($processes) {
            Write-Success "Airflow 服务正在运行:"
            $processes | Format-Table ProcessName, Id, CPU -AutoSize
        } else {
            Write-Warning "Airflow 服务未运行"
        }
    }
    
    "setup" {
        Write-Info "完整设置 Airflow..."
        & $PSCommandPath -Action install
        & $PSCommandPath -Action init
        Write-Success "Airflow 设置完成！使用 './start_airflow_local.ps1 start' 启动服务"
    }
    
    default {
        Write-Info "使用方法:"
        Write-Info "  .\start_airflow_local.ps1 setup     # 完整安装和初始化"
        Write-Info "  .\start_airflow_local.ps1 install   # 仅安装 Airflow"
        Write-Info "  .\start_airflow_local.ps1 init      # 仅初始化数据库"
        Write-Info "  .\start_airflow_local.ps1 start     # 启动服务"
        Write-Info "  .\start_airflow_local.ps1 stop      # 停止服务"
        Write-Info "  .\start_airflow_local.ps1 status    # 查看状态"
    }
}
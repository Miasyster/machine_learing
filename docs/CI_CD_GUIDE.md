# CI/CD指南

本文档详细介绍了机器学习项目的持续集成和持续部署（CI/CD）配置和最佳实践。

## 目录

- [概述](#概述)
- [GitHub Actions配置](#github-actions配置)
- [本地CI工具](#本地ci工具)
- [部署流程](#部署流程)
- [安全配置](#安全配置)
- [监控和告警](#监控和告警)
- [故障排除](#故障排除)

## 概述

我们的CI/CD流程包括以下阶段：

1. **代码质量检查**：语法检查、类型检查、代码风格
2. **测试**：单元测试、集成测试、覆盖率检查
3. **安全扫描**：依赖漏洞检查、代码安全分析
4. **构建**：包构建和验证
5. **部署**：自动化部署到生产环境

## GitHub Actions配置

### 工作流文件

主要配置文件：`.github/workflows/ci.yml`

### 触发条件

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
```

- **Push触发**：main和develop分支的推送
- **PR触发**：针对main分支的Pull Request

### 测试矩阵

```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, "3.10", "3.11"]
```

支持多个Python版本的并行测试。

## 作业详解

### 1. 测试作业 (test)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
```

#### 步骤说明

1. **代码检出**
   ```yaml
   - uses: actions/checkout@v4
   ```

2. **Python环境设置**
   ```yaml
   - name: Set up Python ${{ matrix.python-version }}
     uses: actions/setup-python@v4
     with:
       python-version: ${{ matrix.python-version }}
   ```

3. **依赖缓存**
   ```yaml
   - name: Cache pip dependencies
     uses: actions/cache@v3
     with:
       path: ~/.cache/pip
       key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
   ```

4. **依赖安装**
   ```yaml
   - name: Install dependencies
     run: |
       python -m pip install --upgrade pip
       pip install -r requirements.txt
       pip install -r requirements-dev.txt
   ```

5. **代码检查**
   ```yaml
   - name: Lint with flake8
     run: |
       flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
       flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
   ```

6. **类型检查**
   ```yaml
   - name: Type check with mypy
     run: mypy src/ --ignore-missing-imports
   ```

7. **测试执行**
   ```yaml
   - name: Test with pytest
     run: pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing
   ```

8. **覆盖率上传**
   ```yaml
   - name: Upload coverage to Codecov
     uses: codecov/codecov-action@v3
     with:
       file: ./coverage.xml
       flags: unittests
       name: codecov-umbrella
   ```

### 2. 安全检查作业 (security)

```yaml
security:
  runs-on: ubuntu-latest
  steps:
    - name: Security check with bandit
      run: bandit -r src/
    
    - name: Check dependencies for security vulnerabilities
      run: safety check
```

#### 安全工具

- **Bandit**：Python代码安全漏洞扫描
- **Safety**：依赖包安全漏洞检查

### 3. 构建作业 (build)

```yaml
build:
  needs: [test, security]
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main'
```

#### 构建步骤

1. **包构建**
   ```yaml
   - name: Build package
     run: python -m build
   ```

2. **包验证**
   ```yaml
   - name: Check package
     run: twine check dist/*
   ```

3. **构件上传**
   ```yaml
   - name: Upload artifacts
     uses: actions/upload-artifact@v3
     with:
       name: dist
       path: dist/
   ```

### 4. 部署作业 (deploy)

```yaml
deploy:
  needs: build
  runs-on: ubuntu-latest
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
```

#### 部署步骤

1. **构件下载**
   ```yaml
   - name: Download artifacts
     uses: actions/download-artifact@v3
   ```

2. **PyPI发布**
   ```yaml
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
       skip_existing: true
   ```

## 本地CI工具

### Tox配置

使用tox进行本地多环境测试：

```ini
[tox]
envlist = py38,py39,py310,py311,flake8,mypy,coverage

[testenv]
deps = -r{toxinidir}/requirements-dev.txt
commands = pytest {posargs}

[testenv:flake8]
deps = flake8
commands = flake8 src tests

[testenv:mypy]
deps = mypy
commands = mypy src --ignore-missing-imports

[testenv:coverage]
deps = 
    pytest-cov
    coverage
commands = 
    pytest --cov=src --cov-report=html --cov-report=term-missing
    coverage report --fail-under=80
```

### Pre-commit钩子

配置文件：`.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

#### 安装和使用

```bash
# 安装pre-commit
pip install pre-commit

# 安装钩子
pre-commit install

# 手动运行所有钩子
pre-commit run --all-files
```

### Makefile命令

```makefile
# CI相关命令
.PHONY: ci-local ci-test ci-lint ci-security ci-build

ci-local: install-dev lint type-check test security-check
	@echo "本地CI检查完成"

ci-test:
	pytest tests/ --cov=src --cov-report=xml --cov-report=term-missing

ci-lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

ci-security:
	bandit -r src/
	safety check

ci-build:
	python -m build
	twine check dist/*
```

## 部署流程

### 环境配置

#### 开发环境 (develop分支)
- 自动运行测试
- 不进行部署
- 生成测试报告

#### 生产环境 (main分支)
- 完整CI/CD流程
- 自动部署到PyPI
- 生成发布标签

### 版本管理

使用语义化版本控制：

```
MAJOR.MINOR.PATCH
```

- **MAJOR**：不兼容的API更改
- **MINOR**：向后兼容的功能添加
- **PATCH**：向后兼容的错误修复

### 发布流程

1. **开发完成**：在develop分支完成功能开发
2. **创建PR**：向main分支创建Pull Request
3. **代码审查**：团队成员进行代码审查
4. **CI检查**：自动运行CI流程
5. **合并代码**：审查通过后合并到main
6. **自动部署**：触发自动部署流程
7. **版本标签**：创建版本标签

## 安全配置

### Secrets管理

在GitHub仓库设置中配置以下secrets：

- `PYPI_API_TOKEN`：PyPI发布令牌
- `CODECOV_TOKEN`：Codecov上传令牌

### 权限配置

```yaml
permissions:
  contents: read
  security-events: write
  actions: read
```

### 安全扫描

#### Bandit配置

```yaml
# .bandit
[bandit]
exclude_dirs = tests,docs
skips = B101,B601
```

#### Safety配置

```bash
# 忽略特定漏洞
safety check --ignore 12345
```

## 监控和告警

### 状态徽章

在README.md中添加状态徽章：

```markdown
[![CI](https://github.com/username/repo/workflows/CI/badge.svg)](https://github.com/username/repo/actions)
[![Coverage](https://codecov.io/gh/username/repo/branch/main/graph/badge.svg)](https://codecov.io/gh/username/repo)
[![PyPI](https://img.shields.io/pypi/v/package-name.svg)](https://pypi.org/project/package-name/)
```

### 通知配置

#### Slack通知

```yaml
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    channel: '#ci-cd'
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
  if: always()
```

#### 邮件通知

```yaml
- name: Email Notification
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.MAIL_USERNAME }}
    password: ${{ secrets.MAIL_PASSWORD }}
    subject: CI/CD Status
    body: Build ${{ job.status }}
  if: failure()
```

## 故障排除

### 常见问题

#### 1. 测试失败

```bash
# 本地复现
pytest tests/ -v --tb=long

# 检查特定测试
pytest tests/test_specific.py::test_method -v
```

#### 2. 依赖冲突

```bash
# 检查依赖
pip check

# 更新依赖
pip-compile requirements.in
```

#### 3. 覆盖率不足

```bash
# 生成详细覆盖率报告
pytest --cov=src --cov-report=html
# 查看 htmlcov/index.html
```

#### 4. 安全扫描失败

```bash
# 本地运行安全检查
bandit -r src/
safety check

# 查看详细报告
bandit -r src/ -f json -o bandit-report.json
```

### 调试技巧

#### 1. 启用调试模式

```yaml
- name: Debug
  run: |
    echo "Python version: $(python --version)"
    echo "Pip version: $(pip --version)"
    pip list
```

#### 2. 保存构件

```yaml
- name: Upload logs
  uses: actions/upload-artifact@v3
  with:
    name: logs
    path: |
      *.log
      test-results/
  if: failure()
```

#### 3. SSH调试

```yaml
- name: Setup tmate session
  uses: mxschmitt/action-tmate@v3
  if: failure()
```

### 性能优化

#### 1. 缓存优化

```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/.cache/pre-commit
    key: ${{ runner.os }}-deps-${{ hashFiles('**/requirements*.txt') }}
```

#### 2. 并行执行

```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9, "3.10", "3.11"]
  max-parallel: 4
```

#### 3. 条件执行

```yaml
- name: Skip if no changes
  run: echo "Skipping tests"
  if: github.event_name == 'pull_request' && !contains(github.event.pull_request.changed_files, 'src/')
```

## 最佳实践

### 1. 分支策略

- **main**：生产分支，受保护
- **develop**：开发分支
- **feature/**：功能分支
- **hotfix/**：热修复分支

### 2. 提交规范

使用约定式提交：

```
type(scope): description

feat(auth): add user authentication
fix(api): resolve data validation issue
docs(readme): update installation guide
```

### 3. PR模板

创建 `.github/pull_request_template.md`：

```markdown
## 变更描述
<!-- 描述此PR的变更内容 -->

## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 重构
- [ ] 性能优化

## 测试
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试完成

## 检查清单
- [ ] 代码符合项目规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] CI检查通过
```

### 4. 发布自动化

使用GitHub Releases自动创建发布：

```yaml
- name: Create Release
  uses: actions/create-release@v1
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  with:
    tag_name: ${{ github.ref }}
    release_name: Release ${{ github.ref }}
    draft: false
    prerelease: false
```

## 相关文档

- [测试指南](TESTING_GUIDE.md)
- [部署指南](DEPLOYMENT.md)
- [API参考](API_REFERENCE.md)
- [最佳实践](BEST_PRACTICES.md)
# Makefile for Machine Learning Project

.PHONY: help install install-dev test test-coverage lint format clean docs build deploy

# 默认目标
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  test-coverage - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  docs         - Build documentation"
	@echo "  build        - Build package"
	@echo "  deploy       - Deploy package"
	@echo "  security     - Run security checks"
	@echo "  performance  - Run performance tests"

# 安装依赖
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# 测试
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=80

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-performance:
	pytest tests/performance/ -v --benchmark-only

# 代码质量
lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# 安全检查
security:
	bandit -r src/
	safety check

# 清理
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .tox/

# 文档
docs:
	sphinx-build -b html docs/ docs/_build/html

docs-clean:
	rm -rf docs/_build/

# 构建和部署
build: clean
	python -m build

deploy: build
	twine upload dist/*

deploy-test: build
	twine upload --repository testpypi dist/*

# 开发环境设置
setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# 运行所有检查
check-all: lint security test-coverage
	@echo "All checks passed!"

# 持续集成
ci: install-dev lint security test-coverage
	@echo "CI pipeline completed successfully!"

# 本地开发
dev-test:
	pytest tests/ -v --tb=short

dev-watch:
	pytest-watch tests/ -- -v --tb=short

# 数据库相关（如果需要）
db-init:
	python scripts/init_db.py

db-migrate:
	python scripts/migrate_db.py

# Docker相关
docker-build:
	docker build -t ml-project .

docker-run:
	docker run -p 8000:8000 ml-project

docker-test:
	docker run ml-project pytest tests/

# 性能分析
profile:
	python -m cProfile -o profile.stats scripts/profile_app.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

memory-profile:
	python -m memory_profiler scripts/memory_test.py

# 代码复杂度分析
complexity:
	radon cc src/ -a
	radon mi src/

# 依赖检查
check-deps:
	pip-audit
	safety check

# 更新依赖
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# 发布准备
prepare-release: clean lint security test-coverage docs build
	@echo "Release preparation complete!"

# 快速开始
quickstart: setup-dev
	@echo "Running quick tests..."
	pytest tests/ -x --tb=short
	@echo "Quick start complete!"
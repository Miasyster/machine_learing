-- ML项目数据库初始化脚本
-- 创建机器学习项目所需的数据库表结构

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- 创建模式
CREATE SCHEMA IF NOT EXISTS ml_data;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS ml_monitoring;
CREATE SCHEMA IF NOT EXISTS ml_lineage;

-- 设置搜索路径
SET search_path TO ml_data, ml_models, ml_monitoring, ml_lineage, public;

-- ================================
-- 原始数据表
-- ================================

-- 交易对信息表
CREATE TABLE IF NOT EXISTS ml_data.trading_pairs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    base_asset VARCHAR(10) NOT NULL,
    quote_asset VARCHAR(10) NOT NULL,
    status VARCHAR(20) DEFAULT 'TRADING',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- K线数据表
CREATE TABLE IF NOT EXISTS ml_data.klines (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    quote_volume DECIMAL(20,8) NOT NULL,
    trades_count INTEGER,
    taker_buy_volume DECIMAL(20,8),
    taker_buy_quote_volume DECIMAL(20,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, interval, open_time)
);

-- 订单簿数据表
CREATE TABLE IF NOT EXISTS ml_data.order_book (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    bids JSONB NOT NULL,
    asks JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp)
);

-- 交易数据表
CREATE TABLE IF NOT EXISTS ml_data.trades (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    trade_id BIGINT NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    is_buyer_maker BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, trade_id)
);

-- ================================
-- 特征数据表
-- ================================

-- 技术指标表
CREATE TABLE IF NOT EXISTS ml_data.technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_value DECIMAL(20,8),
    indicator_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, interval, timestamp, indicator_name)
);

-- 特征矩阵表
CREATE TABLE IF NOT EXISTS ml_data.feature_matrix (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    interval VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature_set VARCHAR(50) NOT NULL,
    features JSONB NOT NULL,
    target DECIMAL(20,8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, interval, timestamp, feature_set)
);

-- ================================
-- 模型相关表
-- ================================

-- 模型信息表
CREATE TABLE IF NOT EXISTS ml_models.model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    hyperparameters JSONB,
    training_data_info JSONB,
    performance_metrics JSONB,
    model_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'training',
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
);

-- 模型预测表
CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id BIGSERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models.model_registry(id),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    prediction_value DECIMAL(20,8) NOT NULL,
    confidence_score DECIMAL(5,4),
    prediction_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 模型性能表
CREATE TABLE IF NOT EXISTS ml_models.model_performance (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models.model_registry(id),
    evaluation_date DATE NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    dataset_type VARCHAR(20) NOT NULL, -- train, validation, test
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_id, evaluation_date, metric_name, dataset_type)
);

-- ================================
-- 监控相关表
-- ================================

-- 数据质量监控表
CREATE TABLE IF NOT EXISTS ml_monitoring.data_quality_checks (
    id SERIAL PRIMARY KEY,
    check_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    column_name VARCHAR(100),
    check_type VARCHAR(50) NOT NULL,
    expected_value JSONB,
    actual_value JSONB,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 系统监控表
CREATE TABLE IF NOT EXISTS ml_monitoring.system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 告警记录表
CREATE TABLE IF NOT EXISTS ml_monitoring.alerts (
    id SERIAL PRIMARY KEY,
    alert_name VARCHAR(100) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100)
);

-- ================================
-- 数据血缘表
-- ================================

-- 数据资产表
CREATE TABLE IF NOT EXISTS ml_lineage.data_assets (
    id SERIAL PRIMARY KEY,
    asset_name VARCHAR(200) NOT NULL UNIQUE,
    asset_type VARCHAR(50) NOT NULL,
    schema_name VARCHAR(100),
    table_name VARCHAR(100),
    description TEXT,
    owner VARCHAR(100),
    tags JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 血缘关系表
CREATE TABLE IF NOT EXISTS ml_lineage.lineage_relationships (
    id SERIAL PRIMARY KEY,
    source_asset_id INTEGER REFERENCES ml_lineage.data_assets(id),
    target_asset_id INTEGER REFERENCES ml_lineage.data_assets(id),
    relationship_type VARCHAR(50) NOT NULL,
    transformation_logic TEXT,
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_asset_id, target_asset_id, relationship_type)
);

-- 数据版本表
CREATE TABLE IF NOT EXISTS ml_lineage.data_versions (
    id SERIAL PRIMARY KEY,
    asset_id INTEGER REFERENCES ml_lineage.data_assets(id),
    version_number VARCHAR(20) NOT NULL,
    version_hash VARCHAR(64),
    row_count BIGINT,
    file_size BIGINT,
    checksum VARCHAR(64),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(asset_id, version_number)
);

-- ================================
-- 索引创建
-- ================================

-- K线数据索引
CREATE INDEX IF NOT EXISTS idx_klines_symbol_interval_time ON ml_data.klines(symbol, interval, open_time);
CREATE INDEX IF NOT EXISTS idx_klines_close_time ON ml_data.klines(close_time);

-- 交易数据索引
CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON ml_data.trades(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON ml_data.trades(timestamp);

-- 订单簿索引
CREATE INDEX IF NOT EXISTS idx_order_book_symbol_timestamp ON ml_data.order_book(symbol, timestamp);

-- 技术指标索引
CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_interval_time ON ml_data.technical_indicators(symbol, interval, timestamp);
CREATE INDEX IF NOT EXISTS idx_technical_indicators_name ON ml_data.technical_indicators(indicator_name);

-- 特征矩阵索引
CREATE INDEX IF NOT EXISTS idx_feature_matrix_symbol_interval_time ON ml_data.feature_matrix(symbol, interval, timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_matrix_feature_set ON ml_data.feature_matrix(feature_set);

-- 预测结果索引
CREATE INDEX IF NOT EXISTS idx_predictions_model_symbol_time ON ml_models.predictions(model_id, symbol, timestamp);

-- 监控索引
CREATE INDEX IF NOT EXISTS idx_data_quality_checks_table_time ON ml_monitoring.data_quality_checks(table_name, checked_at);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON ml_monitoring.system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_type_status_time ON ml_monitoring.alerts(alert_type, status, created_at);

-- 血缘索引
CREATE INDEX IF NOT EXISTS idx_lineage_relationships_source ON ml_lineage.lineage_relationships(source_asset_id);
CREATE INDEX IF NOT EXISTS idx_lineage_relationships_target ON ml_lineage.lineage_relationships(target_asset_id);

-- ================================
-- 触发器和函数
-- ================================

-- 更新时间戳函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表添加更新时间戳触发器
CREATE TRIGGER update_trading_pairs_updated_at BEFORE UPDATE ON ml_data.trading_pairs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_model_registry_updated_at BEFORE UPDATE ON ml_models.model_registry FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_data_assets_updated_at BEFORE UPDATE ON ml_lineage.data_assets FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ================================
-- 初始数据插入
-- ================================

-- 插入常用交易对
INSERT INTO ml_data.trading_pairs (symbol, base_asset, quote_asset) VALUES
('BTCUSDT', 'BTC', 'USDT'),
('ETHUSDT', 'ETH', 'USDT'),
('BNBUSDT', 'BNB', 'USDT'),
('ADAUSDT', 'ADA', 'USDT'),
('DOTUSDT', 'DOT', 'USDT'),
('LINKUSDT', 'LINK', 'USDT'),
('LTCUSDT', 'LTC', 'USDT'),
('BCHUSDT', 'BCH', 'USDT'),
('XLMUSDT', 'XLM', 'USDT'),
('EOSUSDT', 'EOS', 'USDT')
ON CONFLICT (symbol) DO NOTHING;

-- 插入基础数据资产
INSERT INTO ml_lineage.data_assets (asset_name, asset_type, schema_name, table_name, description, owner) VALUES
('klines_data', 'table', 'ml_data', 'klines', 'K线历史数据', 'ml_team'),
('trades_data', 'table', 'ml_data', 'trades', '交易历史数据', 'ml_team'),
('order_book_data', 'table', 'ml_data', 'order_book', '订单簿数据', 'ml_team'),
('technical_indicators', 'table', 'ml_data', 'technical_indicators', '技术指标数据', 'ml_team'),
('feature_matrix', 'table', 'ml_data', 'feature_matrix', '特征矩阵数据', 'ml_team'),
('model_predictions', 'table', 'ml_models', 'predictions', '模型预测结果', 'ml_team')
ON CONFLICT (asset_name) DO NOTHING;

-- 创建视图
CREATE OR REPLACE VIEW ml_data.latest_klines AS
SELECT DISTINCT ON (symbol, interval) 
    symbol, interval, open_time, close_time,
    open_price, high_price, low_price, close_price,
    volume, quote_volume
FROM ml_data.klines
ORDER BY symbol, interval, open_time DESC;

CREATE OR REPLACE VIEW ml_models.active_models AS
SELECT * FROM ml_models.model_registry
WHERE status = 'active';

CREATE OR REPLACE VIEW ml_monitoring.recent_alerts AS
SELECT * FROM ml_monitoring.alerts
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY created_at DESC;

-- 创建用户（如果不存在）
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'ml_user') THEN
        CREATE USER ml_user WITH PASSWORD 'ml_password';
    END IF;
END
$$;

-- 授权
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_data TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_models TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_monitoring TO ml_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_lineage TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_data TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_models TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_monitoring TO ml_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_lineage TO ml_user;
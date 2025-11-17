-- Table 1: btc_iv_spd_skewness
CREATE TABLE btc_iv_spd_skewness (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    expiration_timestamp INT NOT NULL,
    atm_slope FLOAT
);

-- Table 2: btc_options_raw
CREATE TABLE btc_options_raw (
    timestamp BIGINT NOT NULL,
    instrument_name VARCHAR(50) NOT NULL,
    expiration_timestamp BIGINT NOT NULL,
    option_type ENUM('call', 'put') NOT NULL,
    bid_price FLOAT,
    ask_price FLOAT,
    bid_iv FLOAT,
    ask_iv FLOAT,
    underlying_price FLOAT,
    strike_price FLOAT,
    log_moneyness FLOAT,
    delta FLOAT,
    vega FLOAT,
    theta FLOAT
);

-- Table 3: btc_options_tick
CREATE TABLE btc_options_tick (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    instrument_name VARCHAR(50) NOT NULL,
    underlying_price FLOAT,
    strike_price FLOAT,
    mid_price FLOAT,
    mark_iv FLOAT,
    expiration_timestamp BIGINT NOT NULL,
    option_type ENUM('call', 'put') NOT NULL,
    log_moneyness FLOAT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 4: btc_historical_futures_prices
-- Stores historical OHLCV data for futures (underlying prices)
CREATE TABLE btc_historical_futures_prices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    instrument_name VARCHAR(50) NOT NULL,
    expiration_timestamp BIGINT,
    open_price FLOAT NOT NULL,
    high_price FLOAT NOT NULL,
    low_price FLOAT NOT NULL,
    close_price FLOAT NOT NULL,
    volume FLOAT DEFAULT 0,
    cost FLOAT DEFAULT 0,
    resolution_seconds INT NOT NULL,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_timestamp_instrument (timestamp, instrument_name),
    INDEX idx_timestamp (timestamp),
    INDEX idx_instrument (instrument_name),
    INDEX idx_expiration (expiration_timestamp),
    INDEX idx_timestamp_instrument (timestamp, instrument_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table 5: btc_historical_option_prices
-- Stores historical OHLCV data for options
CREATE TABLE btc_historical_option_prices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    instrument_name VARCHAR(50) NOT NULL,
    expiration_timestamp BIGINT NOT NULL,
    option_type ENUM('call', 'put') NOT NULL,
    strike_price FLOAT NOT NULL,
    open_price FLOAT NOT NULL,
    high_price FLOAT NOT NULL,
    low_price FLOAT NOT NULL,
    close_price FLOAT NOT NULL,
    volume FLOAT DEFAULT 0,
    cost FLOAT DEFAULT 0,
    resolution_seconds INT NOT NULL,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_timestamp_instrument (timestamp, instrument_name),
    INDEX idx_timestamp (timestamp),
    INDEX idx_instrument (instrument_name),
    INDEX idx_expiration (expiration_timestamp),
    INDEX idx_strike (strike_price),
    INDEX idx_timestamp_expiration (timestamp, expiration_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table 6: btc_historical_volatility_curves
-- Stores historical volatility curves at different timestamps
CREATE TABLE btc_historical_volatility_curves (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    expiration_timestamp BIGINT NOT NULL,
    expiration_str VARCHAR(20) NOT NULL,
    underlying_price FLOAT NOT NULL,
    atm_iv FLOAT NOT NULL,
    atm_slope FLOAT NOT NULL,
    curvature FLOAT,
    num_points INT NOT NULL,
    curve_data JSON,
    -- curve_data stores all volatility curve points as JSON:
    -- [{"strike": 150000, "log_moneyness": -0.05, "iv": 0.85, "option_type": "call"}, ...]
    -- SVI parameters (added for SVI parameterization)
    svi_a FLOAT,
    svi_b FLOAT,
    svi_rho FLOAT,
    svi_m FLOAT,
    svi_sigma FLOAT,
    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_timestamp_expiration (timestamp, expiration_timestamp),
    INDEX idx_timestamp (timestamp),
    INDEX idx_expiration (expiration_timestamp),
    INDEX idx_expiration_str (expiration_str),
    INDEX idx_timestamp_expiration (timestamp, expiration_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

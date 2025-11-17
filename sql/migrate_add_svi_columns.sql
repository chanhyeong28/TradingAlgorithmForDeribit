-- Migration: Add SVI parameter columns to btc_historical_volatility_curves table
-- Run this before using SVI parameterization

ALTER TABLE btc_historical_volatility_curves
ADD COLUMN svi_a FLOAT NULL AFTER curve_data,
ADD COLUMN svi_b FLOAT NULL AFTER svi_a,
ADD COLUMN svi_rho FLOAT NULL AFTER svi_b,
ADD COLUMN svi_m FLOAT NULL AFTER svi_rho,
ADD COLUMN svi_sigma FLOAT NULL AFTER svi_m;


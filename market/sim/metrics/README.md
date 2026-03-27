# Metrics Module

This folder contains market-level and agent-level analytics helpers.

## Purpose

- Compute summary statistics from price, spread, depth, and trade history
- Expose convenience PnL and risk metrics
- Integrate Mesa data collection for step-by-step reporting

## Files

- `market_stats.py`: Midprice, spread, volatility, liquidity depth, order-flow, and market snapshots.
- `pnl.py`: PnL-related helpers such as Sharpe ratio, max drawdown, win rate, and profit factor.
- `datacollector_config.py`: Mesa `DataCollector` wrapper for model and agent metrics.

## Key Outputs

- `MarketStats`: Structured point-in-time market snapshot.
- `PnLMetrics`: Structured profitability summary.
- DataFrames from `MetricsDataCollector` for model and agent histories.

## Typical Usage

- Use `get_market_snapshot(...)` during or after a run for compact market summaries.
- Use `calculate_order_flow(...)` on normalized trade dictionaries.
- Use `MetricsDataCollector` when you need per-step tabular output for analysis.

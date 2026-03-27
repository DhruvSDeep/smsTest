# Visualization Module

This folder contains chart, table, and server-side rendering helpers for market output.

## Purpose

- Format market state for terminal-style inspection
- Build data series for charts
- Provide a server wrapper for interactive visualization flows

## Files

- `charts.py`: Price, spread, and volume chart data helpers.
- `tables.py`: Text tables for trades, leaderboard, order book, and market state.
- `orderbook_view.py`: Visual order-book histogram and best bid/ask text views.
- `server.py`: Visualization server integration layer.

## Typical Usage

- The main `visualise.py` script uses these components to print tables and assemble report data.
- `PriceChart`, `SpreadChart`, and `VolumeChart` store time-series data during a run.
- `OrderBookView` and `TradeTable` are useful for verbose console inspection.

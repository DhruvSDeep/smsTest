# Model Module

This folder contains the top-level simulation orchestration code.

## Purpose

- Own the main market model
- Coordinate the exchange, environment, news, and agent stepping
- Expose market state and leaderboard views for reporting

## Files

- `market_model.py`: Main `MarketModel` class used by scripts and runtime helpers.
- `scheduler_logic.py`: Market environment support, including optional fundamentals and regime logic.

## Core Responsibilities

- Advance the simulation clock
- Update the market environment each tick
- Step every registered trading agent
- Broadcast structured news events
- Aggregate market and agent metrics

## Main Types

- `MarketModel`: Central orchestration object for the simulation.
- `MarketEnvironment`: Optional non-exchange state such as reference price and regimes.
- `MarketRegime`: Named environment state with drift and volatility characteristics.

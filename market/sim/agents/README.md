# Agents Module

This folder contains the trading-agent layer for the simulation.

## Purpose

- Wrap Mesa agents with portfolio state and order submission logic
- Define the strategy contract used by all agent behaviors
- Load built-in or custom strategies at runtime

## Files

- `mesa_agent.py`: Trading agent implementation that owns cash, position, pending orders, fills, and strategy execution.
- `strategy_interface.py`: Shared data contracts and abstract strategy interface.
- `strategy_loader.py`: Strategy registry and dynamic loader for built-in and external strategies.
- `builtins/`: Built-in strategy implementations used by the default configs.

## Main Concepts

- `TradingAgent`: The runtime actor that steps once per tick and submits orders.
- `Observation`: Immutable market snapshot provided to strategies.
- `OrderRequest`: Strategy output describing a desired order.
- `Strategy`: Base interface for all strategy implementations.

## Typical Flow

1. The model builds an `Observation` for each agent.
2. The agent calls `strategy.act(observation)`.
3. Returned `OrderRequest` values are turned into exchange orders.
4. Fills are applied back to the agent portfolio and trade history.

## Related Modules

- `sim.model`: Orchestrates agent stepping and market state updates.
- `sim.exchange`: Processes submitted orders and produces trades.
- `sim.news`: Supplies structured news events to strategies.

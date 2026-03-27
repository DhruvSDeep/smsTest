# Built-in Strategies

This folder contains the default strategy implementations bundled with the simulation.

## Purpose

- Provide out-of-the-box market participants for experimentation
- Cover both passive liquidity provision and aggressive order-taking behavior
- Serve as reference implementations for custom strategies

## Files

- `market_maker.py`: Continuously quotes both sides near the current touch.
- `liquidity_maker.py`: Passive quoting strategy with configurable spread and participation rate.
- `liquidity_taker.py`: Aggressive strategy that crosses the spread using market orders.
- `random_trader.py`: Mixed random strategy used for background activity and testing.

## Strategy Roles

- Market makers keep the book tight and refresh stale quotes.
- Liquidity makers provide resting depth and can lean with directional/news bias.
- Liquidity takers consume liquidity and help move price when flow becomes imbalanced.
- Random traders inject noisy, less structured participation.

## Extension Notes

- All strategies inherit from `sim.agents.strategy_interface.Strategy`.
- Each strategy receives only an `Observation` and returns `OrderRequest` values.
- Strategies can opt into quote refresh by overriding `refresh_orders()`.

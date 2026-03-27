# Exchange Module

This folder contains the matching and order-book layer for the simulation.

## Purpose

- Represent orders, trades, and post-submission order status
- Maintain the live order book
- Match incoming orders using price-time priority

## Files

- `order.py`: Core data structures such as `Order`, `Trade`, `OrderStatus`, `OrderType`, and `Side`.
- `trade.py`: Backwards-compatible re-export of trade-related types.
- `orderbook.py`: Heap-based order book with best bid/ask and depth snapshots.
- `matching_engine.py`: Main order-entry and trade-generation logic.

## Matching Rules

- Bids match the lowest available asks when they cross.
- Asks match the highest available bids when they cross.
- Within a price level, older orders keep priority.
- Market orders consume opposite-side liquidity immediately.

## Public Surface

- `MatchingEngine`: Create orders, submit them, and inspect market state.
- `OrderBook`: Inspect depth, best prices, and active orders.
- `Order`: Internal order representation used by the book and engine.
- `Trade`: Execution record produced by successful matches.
- `OrderStatus`: Result of order processing, including fills and resulting trades.

## Typical Usage

```python
from sim.exchange import MatchingEngine, OrderType, Side

engine = MatchingEngine()
engine.next_tick()

order = engine.create_order(
  agent_id=1,
  side=Side.BID,
  order_type=OrderType.LIMIT,
  quantity=10.0,
  price=100.0,
)
status = engine.submit_order(order)

state = engine.get_market_state()
print(state["best_bid"], state["best_ask"], status.filled_quantity)
```

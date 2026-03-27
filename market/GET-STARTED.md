# Market Simulation Guide

This folder contains the runnable market simulation entry point, its configuration, and the generated report output.

## Main Entry Point

Use `visualise.py` to run a simulation from `config.json`, print a summary, and generate a chart image.

Run it from the `market` folder:

```powershell
python .\visualise.py
```

Or with the explicit interpreter:

```powershell
"C:/Program Files/Python312/python.exe" .\visualise.py
```

What it does:

1. Loads `config.json`
2. Builds the market model and agents
3. Runs the configured number of steps
4. Prints a summary and leaderboard
5. Saves `market_report.png`

## Files In This Folder

- `visualise.py`: Main script for running and visualizing the simulation
- `config.json`: Simulation configuration used by the script
- `market_report.png`: Generated report image after a run
- `sim/`: Core simulation package
- `test/`: Regression tests

## Configuration

`visualise.py` reads `config.json` from the same folder.

### Current Shape

```json
{
  "seed": 4342,
  "max_steps": 500,
  "exchange": {
    "tick_interval": 1.0,
    "max_order_size": 1000.0,
    "min_order_size": 0.01,
    "price_precision": 4
  },
  "market": {
    "initial_price": 100.0,
    "initial_spread": 1.0,
    "enable_fundamentals": false,
    "enable_regimes": false
  },
  "agents": {
    "market_maker": { ... },
    "liquidity_taker": { ... },
    "liquidity_maker": { ... },
    "random_trader": { ... }
  }
}
```

### Top-Level Fields

- `seed`: Reproducible random seed for the run
- `max_steps`: Number of ticks to simulate
- `exchange`: Exchange-level settings
- `market`: Environment and starting-price settings
- `agents`: Strategy groups and their counts/parameters

### Exchange Settings

- `tick_interval`: Logical time step between ticks
- `max_order_size`: Intended cap for order size configuration
- `min_order_size`: Intended floor for order size configuration
- `price_precision`: Intended price display/rounding precision

Note:

The script currently uses `tick_interval` directly. The other exchange keys are useful as configuration metadata and future constraints, but they are not the main tuning knobs for behavior at the moment.

### Market Settings

- `initial_price`: Starting reference price for the simulation
- `initial_spread`: Desired starting spread in config terms
- `enable_fundamentals`: Enables the environment reference-price process
- `enable_regimes`: Enables regime switching in the environment

If you want the simulation to move more from the environment layer instead of mostly from order flow, enable `fundamentals` or `regimes` here.

## How Agents Are Configured

Each key inside `agents` is a strategy name that maps to:

- `count`: Number of agents using that strategy
- `initial_cash`: Starting cash per agent in that group
- `strategy_params`: Constructor parameters passed into the strategy

Example:

```json
"market_maker": {
  "count": 5,
  "initial_cash": 100000.0,
  "strategy_params": {
    "spread": 0.003,
    "order_quantity": 20.0,
    "max_position": 500.0
  }
}
```

## How To Add More Agents

To add more agents of an existing strategy, increase `count`.

Example:

```json
"random_trader": {
  "count": 12,
  "initial_cash": 100000.0,
  "strategy_params": {
    "order_quantity": 20.0,
    "probability": 0.6
  }
}
```

That will create 12 random traders instead of 5.

## How To Add A New Strategy Type

1. Create the strategy implementation in `sim/agents/builtins/` or load it through the strategy loader.
2. Register it in `sim/agents/strategy_loader.py`.
3. Add a new block in `config.json` using the registered strategy name.

Example config entry:

```json
"my_new_strategy": {
  "count": 3,
  "initial_cash": 50000.0,
  "strategy_params": {
    "order_quantity": 10.0,
    "probability": 0.4
  }
}
```

The strategy name in the config must match the name registered in the loader.

## How To Tune Agent Behavior

### Market Makers

Useful parameters:

- `spread`: Quote width around the current market
- `order_quantity`: Quote size
- `max_position`: Inventory limit before the strategy reduces quoting

Use this when you want tighter books and more stable liquidity.

### Liquidity Makers

Useful parameters:

- `spread`: Distance from fair value when placing passive quotes
- `order_quantity`: Size of each quote
- `probability`: How often the strategy participates per tick

Use this when you want more depth without making every order aggressive.

### Liquidity Takers

Useful parameters:

- `order_quantity`: Size of each aggressive order
- `probability`: How often the strategy crosses the spread

Increase this group if price is too static and you want more trade-through behavior.

### Random Traders

Useful parameters:

- `order_quantity`: Size of each order
- `probability`: How often the strategy participates

Use these to add noise and mixed directional flow.

## Reading The Output

After the run, `visualise.py` prints:

1. Progress every 50 steps
2. A simulation summary
3. Final market state
4. Final order book snapshot
5. Recent trades
6. Leaderboard

It also writes `market_report.png`.

## Interpreting Levels And The Order Book

### Best Bid And Best Ask

- Best bid: highest price someone is willing to buy at
- Best ask: lowest price someone is willing to sell at
- Midprice: average of best bid and best ask
- Spread: best ask minus best bid

If the spread is tight, the market is more liquid.
If the spread widens, liquidity is thinner or agents are more cautious.

### Order Book Levels

The table shows price levels and total quantity resting at each level.

Example interpretation:

- Large bid quantity near the top of book means stronger nearby buy support
- Large ask quantity near the top of book means stronger nearby sell resistance
- Balanced depth on both sides usually means a more stable book
- Thin depth means price can move more easily when takers arrive

### Order Book Histogram

This is a visual depth view.

- Longer bars mean more resting quantity at that level
- Bars close to midprice matter most for short-term price movement
- If one side is much deeper than the other, future moves often become easier in the thinner direction

## Interpreting The Report Chart

### Price Panel

- `Midprice`: central market price estimate
- `Best Bid`: top buy quote
- `Best Ask`: top sell quote

If all three lines are moving, the market is repricing actively.

### Spread Panel

- Shows how tight or wide the market is over time

Frequent spikes usually mean liquidity was briefly pulled or the book thinned out.

### Book Volume Panel

- `Bid Volume`: resting buy-side volume
- `Ask Volume`: resting sell-side volume

Rising depth means more passive liquidity is accumulating.
Falling depth usually means takers are consuming the book or makers are pulling quotes.

### Trades And Buy Pressure Panel

- Gray bars: trades per tick
- Red line: buy pressure from recent trade flow

Positive buy pressure means recent aggressive flow is more buyer-driven.
Negative buy pressure means recent aggressive flow is more seller-driven.

## Practical Tuning Tips

If the market is too static:

- Increase `liquidity_taker.probability`
- Increase `random_trader.probability`
- Reduce maker `spread`
- Enable market fundamentals or regimes

If the book gets too thin:

- Increase `market_maker.count`
- Increase `liquidity_maker.count`
- Increase maker `order_quantity`

If price jumps too violently:

- Reduce taker `order_quantity`
- Reduce taker `probability`
- Increase passive maker counts

## Running Tests

Regression tests live under `test/`.

Run them with:

```powershell
python -m unittest discover -s test -p "test_*.py"
```

## Related Documentation

- `sim/agents/README.md`
- `sim/exchange/README.md`
- `sim/metrics/README.md`
- `sim/model/README.md`
- `sim/runtime/README.md`
- `sim/visualization/README.md`
#!/usr/bin/env python3
"""Visualize market simulation.

This script runs a market simulation based on config.json and
visualizes the resulting price data.
"""

import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add the market directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from sim.model.market_model import MarketModel
from sim.agents import get_loader
from sim.exchange import Side
from sim.metrics import (
    calculate_liquidity_depth,
    calculate_max_drawdown,
    calculate_order_flow,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_volatility,
    get_market_snapshot,
)
from sim.news import NewsEvent
from sim.visualization import (
    BestBidAskView,
    TradeTable,
    PriceChart,
    SpreadChart,
    VolumeChart,
    MarketStateTable,
    OrderBookTable,
    OrderBookView,
    LeaderboardTable,
)


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def create_model_from_config(config: Dict) -> MarketModel:
    """Create a MarketModel from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        MarketModel instance
    """
    seed = config.get("seed", 42)
    market_config = config.get("market", {})
    exchange_config = config.get("exchange", {})
    tick_interval = exchange_config.get("tick_interval", 1.0)

    # Build per-commodity configs (up to 5)
    commodity_configs = market_config.get("commodities", [])
    if not commodity_configs:
        # Fallback: single commodity from initial_price
        initial_price = market_config.get("initial_price", 100.0)
        commodity_configs = [{"name": "DEFAULT", "initial_price": initial_price, "initial_spread": 1.0}]

    # Create model with no agents initially
    model = MarketModel(
        seed=seed,
        num_agents=0,
        tick_interval=tick_interval,
        commodity_configs=commodity_configs,
    )

    # Add agents from config (each agent trades all commodities)
    agent_configs = config.get("agents", {})
    strategy_loader = get_loader()

    total_agents = 0
    for strategy_name, agent_config in agent_configs.items():
        count = agent_config.get("count", 0)
        initial_cash = agent_config.get("initial_cash", 10000.0)
        strategy_params = agent_config.get("strategy_params", {})

        for _ in range(count):
            strategy_kwargs = dict(strategy_params)
            strategy_kwargs.setdefault("seed", model.random.randrange(2**32))
            strategy = strategy_loader.create(strategy_name, **strategy_kwargs)

            model.add_agent(
                strategy=strategy,
                initial_cash=initial_cash,
            )
            total_agents += 1

    # Seed all commodity order books with initial orders
    model._initialize_market()

    print(f"Created {total_agents} agents across {len(model.commodities)} commodities: {model.commodities}")
    return model


def build_news_schedule(config: Dict) -> Dict[int, List[NewsEvent]]:
    """Build a tick-indexed news schedule from config."""
    schedule: Dict[int, List[NewsEvent]] = {}
    for raw_event in config.get("news_events", []):
        event = NewsEvent.from_dict(raw_event)
        if event.tick is None:
            continue
        schedule.setdefault(int(event.tick), []).append(event)
    return schedule


def trade_to_dict(trade) -> Dict:
    """Normalize trade objects into dictionaries for reporting."""
    taker_side = getattr(trade, "taker_side", None)
    return {
        "trade_id": trade.trade_id,
        "price": trade.price,
        "quantity": trade.quantity,
        "timestamp": trade.timestamp,
        "maker_agent_id": trade.maker_agent_id,
        "taker_agent_id": trade.taker_agent_id,
        "side": "buy" if taker_side == Side.BID else "sell",
    }


def effective_price(state: Dict) -> Optional[float]:
    """Return the best available price for metrics calculations."""
    midprice = state.get("midprice")
    if midprice is not None:
        return midprice

    environment = state.get("environment", {})
    return environment.get("fundamental") or environment.get("current_price")


def run_simulation(
    model: MarketModel,
    max_steps: int,
    verbose: bool = False,
    news_schedule: Optional[Dict[int, List[NewsEvent]]] = None,
    commodity: Optional[str] = None,
) -> List[Dict]:
    """Run the simulation and collect market data.

    Args:
        model: MarketModel instance
        max_steps: Number of steps to run
        verbose: If True, show detailed visualization tables

    Returns:
        List of market state dictionaries
    """
    commodity = commodity or model.commodities[0]
    market_history = []
    total_trades_count = 0
    trade_history: List[Dict] = []
    price_history: List[float] = []
    news_schedule = news_schedule or {}

    # Initialize visualization components from the project
    price_chart = PriceChart(max_points=max_steps)
    spread_chart = SpreadChart(max_points=max_steps)
    volume_chart = VolumeChart(max_points=max_steps)

    # Optional visualization tables
    market_table = MarketStateTable() if verbose else None
    orderbook_table = OrderBookTable(depth=5) if verbose else None
    leaderboard_table = LeaderboardTable() if verbose else None
    trade_table = TradeTable(max_trades=10) if verbose else None
    best_bid_ask_view = BestBidAskView() if verbose else None
    orderbook_view = OrderBookView(max_levels=5, width=24) if verbose else None

    print(f"Running simulation for {max_steps} steps...")

    for step in range(max_steps):
        for event in news_schedule.get(step + 1, []):
            model.broadcast_news(event)
            print(
                f"  News @ tick {step + 1}: {event.headline or event.event_id} "
                f"(bias={event.directional_bias:+.2f}, vol={event.volatility_bias:+.2f})"
            )

        model.step()

        # Collect market state for the selected commodity
        state = model.get_market_state(commodity)

        # Count trades from this step
        step_trades = [trade_to_dict(trade) for trade in state.get("last_trades", [])]
        total_trades_count += len(step_trades)
        state["trades_count"] = total_trades_count
        state["last_trades"] = step_trades
        trade_history.extend(step_trades)

        price = effective_price(state)
        if price is not None:
            price_history.append(price)

        snapshot = get_market_snapshot(model.exchange, trade_history, price_history, commodity=commodity)
        state["market_stats"] = {
            "tick": snapshot.tick,
            "midprice": snapshot.midprice,
            "spread": snapshot.spread,
            "spread_pct": snapshot.spread_pct,
            "volatility": snapshot.volatility,
            "bid_depth": snapshot.bid_depth,
            "ask_depth": snapshot.ask_depth,
            "trade_count": snapshot.trade_count,
            "trade_volume": snapshot.trade_volume,
        }
        state["order_flow"] = calculate_order_flow(trade_history)
        state["effective_price"] = price

        market_history.append(state)

        # Update visualization charts
        price_chart.add_point(
            model.tick,
            state.get("midprice"),
            state.get("best_bid"),
            state.get("best_ask"),
        )
        spread_chart.add_point(model.tick, state.get("spread"))
        volume_chart.add_point(
            model.tick,
            state.get("bid_volume", 0),
            state.get("ask_volume", 0),
        )

        # Print progress
        if (step + 1) % 50 == 0:
            midprice = state.get("midprice")
            spread = state.get("spread")
            volatility = state["market_stats"].get("volatility")
            midprice_str = f"{midprice:.2f}" if midprice is not None else "N/A"
            spread_str = f"{spread:.2f}" if spread is not None else "N/A"
            volatility_str = f"{volatility:.4f}" if volatility is not None else "N/A"
            print(
                f"  Step {step + 1}/{max_steps}: midprice={midprice_str}, "
                f"spread={spread_str}, vol={volatility_str}, total_trades={total_trades_count}"
            )

            # Show visualization tables in verbose mode
            if verbose:
                depth = model.exchange.get_depth_snapshot(commodity, depth=5)
                print("\n--- Market State ---")
                print(market_table.format(state))
                print("\n--- Best Bid / Ask ---")
                print(
                    best_bid_ask_view.format(
                        state.get("best_bid"),
                        state.get("best_ask"),
                        state.get("spread"),
                    )
                )
                print("\n--- Order Book ---")
                print(orderbook_table.format(depth["bids"], depth["asks"]))
                print("\n--- Order Book Histogram ---")
                print(orderbook_view.format_histogram(depth["bids"], depth["asks"]))
                print("\n--- Recent Trades ---")
                print(trade_table.format(trade_history))
                print("\n--- Leaderboard ---")
                print(leaderboard_table.format(model.get_leaderboard()))

    # Store charts in model for potential later use
    model._visualization_data = {
        "price_chart": price_chart.get_data(),
        "spread_chart": spread_chart.get_data(),
        "volume_chart": volume_chart.get_data(),
        "trade_history": trade_history,
        "price_history": price_history,
    }

    return market_history


def visualize_market(
    market_history: List[Dict], output_path: Optional[str] = None
) -> None:
    """Visualize market data using a richer multi-panel matplotlib report.

    Args:
        market_history: List of market states
        output_path: Optional path to save the chart
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt

    # Extract data
    ticks = [s["tick"] for s in market_history]
    midprices = [s.get("effective_price") for s in market_history]
    best_bids = [s.get("best_bid") for s in market_history]
    best_asks = [s.get("best_ask") for s in market_history]
    spreads = [s.get("spread") or 0.0 for s in market_history]
    bid_volumes = [s.get("bid_volume", 0.0) for s in market_history]
    ask_volumes = [s.get("ask_volume", 0.0) for s in market_history]
    trade_counts = [len(s.get("last_trades", [])) for s in market_history]
    buy_pressure = [
        s.get("order_flow", {}).get("buy_pressure", 0.0) for s in market_history
    ]

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    price_ax, spread_ax, volume_ax, flow_ax = axes

    # Plot prices
    price_ax.plot(ticks, midprices, label="Midprice", linewidth=1.6, color="blue")
    price_ax.plot(
        ticks, best_bids, label="Best Bid", linewidth=1.0, color="green", alpha=0.7
    )
    price_ax.plot(
        ticks, best_asks, label="Best Ask", linewidth=1.0, color="red", alpha=0.7
    )
    commodity_label = market_history[0].get("commodity", "") if market_history else ""
    price_ax.set_ylabel("Price", fontsize=11)
    price_ax.set_title(f"Market Simulation Report — {commodity_label}", fontsize=14)
    price_ax.legend(loc="upper left")
    price_ax.grid(True, alpha=0.3)

    spread_ax.plot(ticks, spreads, label="Spread", color="purple", linewidth=1.2)
    spread_ax.set_ylabel("Spread", fontsize=11)
    spread_ax.legend(loc="upper left")
    spread_ax.grid(True, alpha=0.3)

    volume_ax.plot(ticks, bid_volumes, label="Bid Volume", color="teal", linewidth=1.1)
    volume_ax.plot(
        ticks, ask_volumes, label="Ask Volume", color="orange", linewidth=1.1
    )
    volume_ax.set_ylabel("Book Volume", fontsize=11)
    volume_ax.legend(loc="upper left")
    volume_ax.grid(True, alpha=0.3)

    flow_ax.bar(ticks, trade_counts, label="Trades / Tick", color="gray", alpha=0.4)
    flow_ax2 = flow_ax.twinx()
    flow_ax2.plot(
        ticks,
        buy_pressure,
        label="Buy Pressure",
        color="brown",
        linewidth=1.2,
    )
    flow_ax.set_ylabel("Trades", fontsize=11)
    flow_ax2.set_ylabel("Buy Pressure", fontsize=11)
    flow_ax.set_xlabel("Tick", fontsize=12)
    flow_ax.grid(True, alpha=0.3)

    flow_lines, flow_labels = flow_ax.get_legend_handles_labels()
    flow2_lines, flow2_labels = flow_ax2.get_legend_handles_labels()
    flow_ax.legend(
        flow_lines + flow2_lines, flow_labels + flow2_labels, loc="upper left"
    )

    # Add some statistics
    if midprices:
        valid_prices = [p for p in midprices if p is not None]
        if valid_prices:
            min_price = min(valid_prices)
            max_price = max(valid_prices)
            avg_price = sum(valid_prices) / len(valid_prices)
            returns = calculate_returns(valid_prices)
            sharpe = calculate_sharpe_ratio(returns)
            drawdown = calculate_max_drawdown(valid_prices)

            stats_text = (
                f"Min: {min_price:.2f}\n"
                f"Max: {max_price:.2f}\n"
                f"Avg: {avg_price:.2f}\n"
                f"Sharpe: {sharpe:.3f}\n"
                f"Max DD: {drawdown:.2f}%"
            )
            price_ax.text(
                0.02,
                0.98,
                stats_text,
                transform=price_ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Chart saved to {output_path}")
    else:
        plt.show()


def print_summary(model: MarketModel, market_history: List[Dict], commodity: Optional[str] = None) -> None:
    """Print a richer simulation report using metrics and visualization helpers."""
    if not market_history:
        print("No data to summarize")
        return

    commodity = commodity or model.commodities[0]

    # Get final state
    final = market_history[-1]
    trade_history = model._visualization_data.get("trade_history", [])
    price_history = model._visualization_data.get("price_history", [])
    leaderboard = model.get_leaderboard()
    depth = model.exchange.get_depth_snapshot(commodity, depth=5)
    snapshot = get_market_snapshot(model.exchange, trade_history, price_history, commodity=commodity)
    order_flow = calculate_order_flow(trade_history)
    returns = calculate_returns(price_history)
    volatility = calculate_volatility(price_history)
    sharpe = calculate_sharpe_ratio(returns)
    max_drawdown = calculate_max_drawdown(price_history)
    bid_depth, ask_depth = calculate_liquidity_depth(
        depth["bids"], depth["asks"], levels=5
    )
    spreads = [s.get("spread") for s in market_history if s.get("spread") is not None]
    recent_news = final.get("news")

    market_table = MarketStateTable()
    best_bid_ask_view = BestBidAskView()
    orderbook_table = OrderBookTable(depth=5)
    orderbook_view = OrderBookView(max_levels=5, width=24)
    trade_table = TradeTable(max_trades=10)
    leaderboard_table = LeaderboardTable(max_agents=10)

    print("\n" + "=" * 50)
    print("SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Total ticks: {final.get('tick', 0)}")

    midprice = final.get("midprice")
    spread = final.get("spread")
    if midprice is not None:
        print(f"Final midprice: {midprice:.2f}")
    else:
        print("Final midprice: N/A")
    if spread is not None:
        print(f"Final spread: {spread:.2f}")
    else:
        print("Final spread: N/A")
    print(f"Total trades: {final.get('trades_count', 0)}")
    print(f"Trade volume: {snapshot.trade_volume:.2f}")
    print(f"Top-5 bid depth: {bid_depth:.2f}")
    print(f"Top-5 ask depth: {ask_depth:.2f}")

    if price_history:
        print(f"Min price: {min(price_history):.2f}")
        print(f"Max price: {max(price_history):.2f}")
        print(f"Average price: {statistics.mean(price_history):.2f}")

    if spreads:
        print(f"Average spread: {statistics.mean(spreads):.4f}")
        print(f"Min spread: {min(spreads):.4f}")
        print(f"Max spread: {max(spreads):.4f}")

    if snapshot.spread_pct is not None:
        print(f"Final spread %: {snapshot.spread_pct:.4f}%")

    if volatility is not None:
        print(f"Rolling volatility: {volatility:.4f}")

    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Max drawdown: {max_drawdown:.2f}%")
    print(f"Buy volume: {order_flow['buy_volume']:.2f}")
    print(f"Sell volume: {order_flow['sell_volume']:.2f}")
    print(f"Buy pressure: {order_flow['buy_pressure']:.4f}")

    if recent_news is not None:
        print(
            f"Latest news: {recent_news.get('headline') or recent_news.get('event_id')}"
        )

    print("=" * 50)

    print("\n--- Final Market State ---")
    print(market_table.format(final))
    print("\n--- Best Bid / Ask ---")
    print(
        best_bid_ask_view.format(
            final.get("best_bid"),
            final.get("best_ask"),
            final.get("spread"),
        )
    )
    print("\n--- Final Order Book ---")
    print(orderbook_table.format(depth["bids"], depth["asks"]))
    print("\n--- Order Book Histogram ---")
    print(orderbook_view.format_histogram(depth["bids"], depth["asks"]))
    print("\n--- Recent Trades ---")
    print(trade_table.format(trade_history))
    print("\n--- Leaderboard ---")
    print(leaderboard_table.format(leaderboard))


def main():
    """Main entry point.

    Usage:
        python visualise.py [commodity]

    Args:
        commodity: Optional commodity name to visualize (e.g. WHEAT, CRUDE_OIL).
                   Defaults to the first commodity in config.
    """
    # Optional commodity argument
    selected_commodity: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None

    # Determine config path
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    # Load config
    print(f"Loading config from {config_path}")
    config = load_config(str(config_path))

    # Create model
    print("Creating market model...")
    model = create_model_from_config(config)

    # Validate commodity selection
    if selected_commodity is None:
        selected_commodity = model.commodities[0]
    elif selected_commodity not in model.commodities:
        print(f"Unknown commodity '{selected_commodity}'. Available: {model.commodities}")
        sys.exit(1)

    print(f"Visualizing commodity: {selected_commodity}")

    # Run simulation
    max_steps = config.get("max_steps", 500)
    news_schedule = build_news_schedule(config)
    market_history = run_simulation(
        model, max_steps, news_schedule=news_schedule, commodity=selected_commodity
    )

    # Print summary
    print_summary(model, market_history, commodity=selected_commodity)

    # Visualize
    print("\nGenerating market report chart...")
    output_path = Path(__file__).parent / "market_report.png"
    visualize_market(market_history, str(output_path))

    print("\nDone!")


if __name__ == "__main__":
    main()

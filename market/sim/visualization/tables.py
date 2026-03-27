"""Tables for displaying market data.

This module provides table components for displaying
order book, trades, and agent leaderboard data.
"""

from typing import Dict, List, Optional


class OrderBookTable:
    """Order book depth table.

    Displays top N levels of bids and asks.
    """

    def __init__(self, depth: int = 10):
        """Initialize table.

        Args:
            depth: Number of price levels to show
        """
        self.depth = depth

    def format(self, bids: List[tuple], asks: List[tuple]) -> str:
        """Format order book as text table.

        Args:
            bids: List of (price, quantity) for bids
            asks: List of (price, quantity) for asks

        Returns:
            Formatted table string
        """
        lines = ["=== Order Book ===", ""]

        # Header
        lines.append(
            f"{'Bid Price':>12} | {'Bid Qty':>10} | {'Ask Qty':>10} | {'Ask Price':>12}"
        )
        lines.append("-" * 50)

        # Get max rows
        max_rows = max(len(bids), len(asks), self.depth)

        for i in range(min(max_rows, self.depth)):
            bid_price = ""
            bid_qty = ""
            ask_price = ""
            ask_qty = ""

            if i < len(bids):
                bid_price = f"{bids[i][0]:.2f}"
                bid_qty = f"{bids[i][1]:.0f}"

            if i < len(asks):
                ask_price = f"{asks[i][0]:.2f}"
                ask_qty = f"{asks[i][1]:.0f}"

            lines.append(
                f"{bid_price:>12} | {bid_qty:>10} | {ask_qty:>10} | {ask_price:>12}"
            )

        return "\n".join(lines)


class TradeTable:
    """Trade history table.

    Displays recent trades.
    """

    def __init__(self, max_trades: int = 20):
        """Initialize table.

        Args:
            max_trades: Maximum trades to display
        """
        self.max_trades = max_trades

    def format(self, trades: List[Dict]) -> str:
        """Format trades as text table.

        Args:
            trades: List of trade dictionaries

        Returns:
            Formatted table string
        """
        if not trades:
            return "=== Trades ===\n\n(No trades)"

        lines = ["=== Recent Trades ===", ""]

        # Header
        lines.append(
            f"{'Time':>6} | {'Price':>10} | {'Qty':>8} | {'Maker':>6} | {'Taker':>6}"
        )
        lines.append("-" * 50)

        # Show recent trades (in reverse order - newest first)
        for trade in reversed(trades[-self.max_trades :]):
            tick = int(trade.get("timestamp", 0))
            price = trade.get("price", 0)
            qty = trade.get("quantity", 0)
            maker = trade.get("maker_agent_id", "?")
            taker = trade.get("taker_agent_id", "?")

            lines.append(
                f"{tick:>6} | {price:>10.2f} | {qty:>8.2f} | {maker:>6} | {taker:>6}"
            )

        return "\n".join(lines)


class LeaderboardTable:
    """Agent leaderboard table.

    Displays agents sorted by PnL.
    """

    def __init__(self, max_agents: int = 10):
        """Initialize table.

        Args:
            max_agents: Maximum agents to display
        """
        self.max_agents = max_agents

    def format(self, leaderboard: List[Dict]) -> str:
        """Format leaderboard as text table.

        Args:
            leaderboard: List of agent state dicts sorted by PnL

        Returns:
            Formatted table string
        """
        if not leaderboard:
            return "=== Leaderboard ===\n\n(No agents)"

        lines = ["=== Leaderboard ===", ""]

        # Header
        lines.append(
            f"{'Rank':>4} | {'Agent':>6} | {'Cash':>12} | {'Position':>10} | {'PnL':>12} | {'Return':>8}"
        )
        lines.append("-" * 70)

        for rank, agent in enumerate(leaderboard[: self.max_agents], 1):
            agent_id = agent.get("agent_id", "?")
            cash = agent.get("cash", 0)
            position = agent.get("net_position", 0)
            pnl = agent.get("total_pnl", 0)
            ret_pct = agent.get("return_pct", 0)

            lines.append(
                f"{rank:>4} | {agent_id:>6} | {cash:>12.2f} | {position:>10.2f} | "
                f"{pnl:>12.2f} | {ret_pct:>7.2f}%"
            )

        return "\n".join(lines)


class MarketStateTable:
    """Market state summary table.

    Displays current market state.
    """

    def format(self, market_state: Dict) -> str:
        """Format market state as text.

        Args:
            market_state: Market state dictionary

        Returns:
            Formatted string
        """
        lines = ["=== Market State ===", ""]

        lines.append(f"Tick:        {market_state.get('tick', 'N/A')}")
        lines.append(f"Best Bid:    {market_state.get('best_bid', 'N/A')}")
        lines.append(f"Best Ask:    {market_state.get('best_ask', 'N/A')}")
        lines.append(f"Midprice:    {market_state.get('midprice', 'N/A')}")
        lines.append(f"Spread:      {market_state.get('spread', 'N/A')}")
        lines.append(f"Volume:      {market_state.get('volume', 'N/A')}")
        lines.append(f"Order Count: {market_state.get('order_count', 'N/A')}")
        lines.append(f"Agents:      {market_state.get('num_agents', 'N/A')}")

        return "\n".join(lines)


def format_table(data: List[Dict], columns: List[str]) -> str:
    """Generic table formatter.

    Args:
        data: List of row dictionaries
        columns: Column names to display

    Returns:
        Formatted table string
    """
    if not data:
        return "No data"

    lines = []

    # Header
    header = " | ".join(f"{col:>12}" for col in columns)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for row in data:
        values = []
        for col in columns:
            val = row.get(col, "N/A")
            if isinstance(val, float):
                values.append(f"{val:>12.2f}")
            elif isinstance(val, int):
                values.append(f"{val:>12}")
            else:
                values.append(f"{str(val):>12}")
        lines.append(" | ".join(values))

    return "\n".join(lines)

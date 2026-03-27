"""PnL and profitability metrics.

This module provides functions for calculating agent PnL metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PnLMetrics:
    """PnL metrics for an agent.

    Attributes:
        realized_pnl: Realized profit/loss (closed positions)
        unrealized_pnl: Unrealized profit/loss (open positions)
        total_pnl: Total PnL
        gross_volume: Total trading volume
        net_volume: Net trading volume (buys - sells)
        trade_count: Number of trades
        avg_trade_size: Average trade size
    """

    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    gross_volume: float
    net_volume: float
    trade_count: int
    avg_trade_size: float


def calculate_pnl(
    cash: float,
    initial_cash: float,
    position: float,
    current_price: float,
    avg_cost: Optional[float] = None,
) -> PnLMetrics:
    """Calculate PnL metrics.

    Args:
        cash: Current cash balance
        initial_cash: Starting cash
        position: Current position (positive = long)
        current_price: Current market price
        avg_cost: Average cost basis (optional)

    Returns:
        PnLMetrics
    """
    # Realized PnL is tracked in the agent
    # For this simple version, we calculate total PnL

    # Unrealized PnL based on current position
    if position != 0 and avg_cost is not None:
        unrealized = position * (current_price - avg_cost)
    else:
        unrealized = 0.0

    # Total equity
    equity = cash + position * current_price

    # Total PnL
    total_pnl = equity - initial_cash

    return PnLMetrics(
        realized_pnl=total_pnl - unrealized,
        unrealized_pnl=unrealized,
        total_pnl=total_pnl,
        gross_volume=0.0,  # Would need to track from trades
        net_volume=0.0,
        trade_count=0,
        avg_trade_size=0.0,
    )


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate Sharpe ratio.

    Args:
        returns: List of periodic returns
        risk_free_rate: Risk-free rate (annual)

    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    import statistics

    mean_return = statistics.mean(returns)
    std_return = statistics.stdev(returns)

    if std_return == 0:
        return 0.0

    return (mean_return - risk_free_rate) / std_return


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown.

    Args:
        equity_curve: List of equity values over time

    Returns:
        Maximum drawdown as a percentage
    """
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return max_dd * 100  # As percentage


def calculate_win_rate(trades: List[dict]) -> float:
    """Calculate win rate from trades.

    Args:
        trades: List of trade records

    Returns:
        Win rate as percentage
    """
    if not trades:
        return 0.0

    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return (wins / len(trades)) * 100


def calculate_profit_factor(trades: List[dict]) -> float:
    """Calculate profit factor (gross profits / gross losses).

    Args:
        trades: List of trade records

    Returns:
        Profit factor
    """
    gross_profit = sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss

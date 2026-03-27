"""Market statistics and metrics.

This module provides functions for calculating market-level metrics
like price series, spread, volatility, volume, and liquidity depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class MarketStats:
    """Market statistics snapshot.

    Attributes:
        tick: Current tick
        midprice: Current midprice
        spread: Current spread
        spread_pct: Spread as percentage of midprice
        volatility: Rolling volatility
        bid_depth: Total bid volume at top N levels
        ask_depth: Total ask volume at top N levels
        trade_count: Number of trades in period
        trade_volume: Total trade volume in period
    """

    tick: float
    midprice: Optional[float]
    spread: Optional[float]
    spread_pct: Optional[float]
    volatility: Optional[float]
    bid_depth: float
    ask_depth: float
    trade_count: int
    trade_volume: float


def calculate_midprice(
    best_bid: Optional[float], best_ask: Optional[float]
) -> Optional[float]:
    """Calculate midprice from best bid and ask.

    Args:
        best_bid: Best bid price
        best_ask: Best ask price

    Returns:
        Midprice or None if no quotes
    """
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2


def calculate_spread_pct(
    spread: Optional[float], midprice: Optional[float]
) -> Optional[float]:
    """Calculate spread as percentage of midprice.

    Args:
        spread: Absolute spread
        midprice: Midprice

    Returns:
        Spread percentage or None
    """
    if spread is None or midprice is None or midprice == 0:
        return None
    return (spread / midprice) * 100


def calculate_volatility(prices: List[float], window: int = 20) -> Optional[float]:
    """Calculate rolling volatility.

    Args:
        prices: List of prices
        window: Window size

    Returns:
        Volatility (standard deviation) or None
    """
    if len(prices) < window:
        return None

    recent = prices[-window:]
    mean = sum(recent) / len(recent)

    variance = sum((p - mean) ** 2 for p in recent) / len(recent)
    return math.sqrt(variance)


def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate period-over-period returns.

    Args:
        prices: List of prices

    Returns:
        List of returns
    """
    if len(prices) < 2:
        return []

    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] != 0:
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            returns.append(ret)

    return returns


def calculate_liquidity_depth(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    levels: int = 5,
) -> Tuple[float, float]:
    """Calculate liquidity depth.

    Args:
        bids: List of (price, quantity) for bids
        asks: List of (price, quantity) for asks
        levels: Number of levels to include

    Returns:
        (bid_depth, ask_depth) tuple
    """
    bid_vol = sum(qty for _, qty in bids[:levels])
    ask_vol = sum(qty for _, qty in asks[:levels])

    return bid_vol, ask_vol


def calculate_order_flow(
    trade_list: List[Dict],
    window: int = 100,
) -> Dict:
    """Calculate order flow metrics.

    Args:
        trade_list: List of trade records
        window: Window for calculations

    Returns:
        Dictionary with order flow metrics
    """
    if not trade_list:
        return {
            "buy_volume": 0,
            "sell_volume": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "buy_pressure": 0,
        }

    recent = trade_list[-window:]

    buy_volume = sum(t.get("quantity", 0) for t in recent if t.get("side") == "buy")
    sell_volume = sum(t.get("quantity", 0) for t in recent if t.get("side") == "sell")
    buy_trades = sum(1 for t in recent if t.get("side") == "buy")
    sell_trades = sum(1 for t in recent if t.get("side") == "sell")

    total_volume = buy_volume + sell_volume
    buy_pressure = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0

    return {
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "buy_pressure": buy_pressure,
    }


def get_market_snapshot(
    exchange: "MatchingEngine",
    trade_history: List[Dict],
    price_history: List[float],
    commodity: Optional[str] = None,
) -> MarketStats:
    """Get comprehensive market statistics.

    Args:
        exchange: The matching engine
        trade_history: Recent trade history
        price_history: Price history for volatility
        commodity: Commodity to snapshot (defaults to first commodity)

    Returns:
        MarketStats snapshot
    """
    commodity = commodity or exchange.commodities[0]
    market_state = exchange.get_market_state(commodity)
    depth = exchange.get_depth_snapshot(commodity)

    # Get metrics
    best_bid = market_state["best_bid"]
    best_ask = market_state["best_ask"]
    midprice = calculate_midprice(best_bid, best_ask)
    spread = market_state["spread"]
    spread_pct = calculate_spread_pct(spread, midprice)
    volatility = calculate_volatility(price_history)

    # Liquidity depth
    bid_depth, ask_depth = calculate_liquidity_depth(depth["bids"], depth["asks"])

    # Trade metrics
    trade_count = len(trade_history)
    trade_volume = sum(t.get("quantity", 0) for t in trade_history)

    return MarketStats(
        tick=market_state["tick"],
        midprice=midprice,
        spread=spread,
        spread_pct=spread_pct,
        volatility=volatility,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        trade_count=trade_count,
        trade_volume=trade_volume,
    )


# Type hint
from ..exchange import MatchingEngine

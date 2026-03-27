"""Charts for market visualization.

This module provides chart components for visualizing
price, spread, and volume data.
"""

from typing import Dict, List, Optional, Tuple

# Using simple text-based charts for compatibility
# In a real implementation, this would use a charting library


class PriceChart:
    """Price line chart data generator.

    Generates data for price line charts including
    midprice, best bid, best ask over time.
    """

    def __init__(self, max_points: int = 100):
        """Initialize chart.

        Args:
            max_points: Maximum data points to retain
        """
        self.max_points = max_points
        self.ticks: List[float] = []
        self.midprices: List[Optional[float]] = []
        self.best_bids: List[Optional[float]] = []
        self.best_asks: List[Optional[float]] = []

    def add_point(
        self,
        tick: float,
        midprice: Optional[float],
        best_bid: Optional[float] = None,
        best_ask: Optional[float] = None,
    ) -> None:
        """Add a data point.

        Args:
            tick: Current tick
            midprice: Midprice
            best_bid: Best bid
            best_ask: Best ask
        """
        self.ticks.append(tick)
        self.midprices.append(midprice)
        self.best_bids.append(best_bid)
        self.best_asks.append(best_ask)

        # Trim if needed
        if len(self.ticks) > self.max_points:
            self.ticks = self.ticks[-self.max_points :]
            self.midprices = self.midprices[-self.max_points :]
            self.best_bids = self.best_bids[-self.max_points :]
            self.best_asks = self.best_asks[-self.max_points :]

    def get_data(self) -> Dict:
        """Get chart data.

        Returns:
            Dictionary with chart data
        """
        return {
            "ticks": self.ticks,
            "midprice": self.midprices,
            "best_bid": self.best_bids,
            "best_ask": self.best_asks,
        }

    def get_min_max(self) -> Tuple[float, float]:
        """Get min and max prices for scaling.

        Returns:
            (min_price, max_price) tuple
        """
        all_prices = [
            p for p in self.midprices + self.best_bids + self.best_asks if p is not None
        ]
        if not all_prices:
            return 0.0, 100.0
        return min(all_prices), max(all_prices)


class SpreadChart:
    """Spread bar chart data generator."""

    def __init__(self, max_points: int = 100):
        """Initialize chart.

        Args:
            max_points: Maximum data points
        """
        self.max_points = max_points
        self.ticks: List[float] = []
        self.spreads: List[Optional[float]] = []

    def add_point(self, tick: float, spread: Optional[float]) -> None:
        """Add a data point."""
        self.ticks.append(tick)
        self.spreads.append(spread)

        if len(self.ticks) > self.max_points:
            self.ticks = self.ticks[-self.max_points :]
            self.spreads = self.spreads[-self.max_points :]

    def get_data(self) -> Dict:
        """Get chart data."""
        return {
            "ticks": self.ticks,
            "spread": self.spreads,
        }


class VolumeChart:
    """Volume bar chart data generator."""

    def __init__(self, max_points: int = 100):
        """Initialize chart.

        Args:
            max_points: Maximum data points
        """
        self.max_points = max_points
        self.ticks: List[float] = []
        self.bid_volumes: List[float] = []
        self.ask_volumes: List[float] = []

    def add_point(
        self,
        tick: float,
        bid_volume: float,
        ask_volume: float,
    ) -> None:
        """Add a data point."""
        self.ticks.append(tick)
        self.bid_volumes.append(bid_volume)
        self.ask_volumes.append(ask_volume)

        if len(self.ticks) > self.max_points:
            self.ticks = self.ticks[-self.max_points :]
            self.bid_volumes = self.bid_volumes[-self.max_points :]
            self.ask_volumes = self.ask_volumes[-self.max_points :]

    def get_data(self) -> Dict:
        """Get chart data."""
        return {
            "ticks": self.ticks,
            "bid_volume": self.bid_volumes,
            "ask_volume": self.ask_volumes,
        }


def format_price(price: Optional[float], precision: int = 2) -> str:
    """Format price for display.

    Args:
        price: Price value
        precision: Decimal precision

    Returns:
        Formatted price string
    """
    if price is None:
        return "N/A"
    return f"{price:.{precision}f}"


def format_volume(volume: float) -> str:
    """Format volume for display.

    Args:
        volume: Volume value

    Returns:
        Formatted volume string
    """
    if volume >= 1000:
        return f"{volume/1000:.1f}K"
    return f"{volume:.0f}"

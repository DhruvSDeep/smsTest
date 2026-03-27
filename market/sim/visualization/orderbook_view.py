"""Order book depth visualization.

This module provides visual components for displaying
order book depth as histogram/visualization.
"""

from typing import Dict, List, Optional, Tuple


class OrderBookView:
    """Order book depth histogram visualization.

    Displays order book depth as a visual histogram.
    """

    def __init__(self, max_levels: int = 15, width: int = 40):
        """Initialize view.

        Args:
            max_levels: Maximum price levels to display
            width: Character width for bars
        """
        self.max_levels = max_levels
        self.width = width

    def get_depth_bars(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> List[Dict]:
        """Get depth bars data.

        Args:
            bids: List of (price, quantity) for bids
            asks: List of (price, quantity) for asks

        Returns:
            List of bar dictionaries
        """
        # Find max volume for scaling
        max_vol = 0.0
        for _, qty in bids[: self.max_levels]:
            max_vol = max(max_vol, qty)
        for _, qty in asks[: self.max_levels]:
            max_vol = max(max_vol, qty)

        if max_vol == 0:
            return []

        bars = []

        # Process bids (right to left - highest price first)
        for price, qty in bids[: self.max_levels]:
            bar_width = int((qty / max_vol) * self.width)
            bars.append(
                {
                    "side": "bid",
                    "price": price,
                    "quantity": qty,
                    "bar": "#" * bar_width + "." * (self.width - bar_width),
                }
            )

        # Process asks (left to right - lowest price first)
        for price, qty in asks[: self.max_levels]:
            bar_width = int((qty / max_vol) * self.width)
            bars.append(
                {
                    "side": "ask",
                    "price": price,
                    "quantity": qty,
                    "bar": "#" * bar_width + "." * (self.width - bar_width),
                }
            )

        return bars

    def format_histogram(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
    ) -> str:
        """Format order book as ASCII histogram.

        Args:
            bids: List of (price, quantity) for bids
            asks: List of (price, quantity) for asks

        Returns:
            Formatted histogram string
        """
        bars = self.get_depth_bars(bids, asks)

        if not bars:
            return "=== Order Book Depth ===\n\n(No orders)"

        lines = ["=== Order Book Depth ===", ""]

        # Find midprice for centering
        mid_bid = bids[0][0] if bids else 0
        mid_ask = asks[0][0] if asks else 0
        midprice = (mid_bid + mid_ask) / 2 if bids and asks else mid_bid

        # Center indicator
        lines.append(f"Midprice: {midprice:.2f}")
        lines.append("")

        # Show asks first (lower prices = below midprice)
        lines.append("--- ASKS (Sells) ---")
        for bar in bars:
            if bar["side"] == "ask":
                lines.append(
                    f"{bar['price']:>8.2f} | {bar['bar']} | {bar['quantity']:>8.2f}"
                )

        lines.append(" " * 10 + "+" + "-" * (self.width + 2) + "+")
        lines.append(" " * 10 + "|    MIDPRICE    |")

        # Then bids (higher prices = above midprice)
        lines.append("--- BIDS (Buys) ---")
        for bar in bars:
            if bar["side"] == "bid":
                lines.append(
                    f"{bar['price']:>8.2f} | {bar['bar']} | {bar['quantity']:>8.2f}"
                )

        return "\n".join(lines)


class BestBidAskView:
    """Simple best bid/ask display."""

    def format(
        self,
        best_bid: Optional[float],
        best_ask: Optional[float],
        spread: Optional[float],
    ) -> str:
        """Format best bid/ask display.

        Args:
            best_bid: Best bid price
            best_ask: Best ask price
            spread: Bid-ask spread

        Returns:
            Formatted string
        """
        lines = ["=== Best Bid/Ask ===", ""]

        if best_bid is not None:
            lines.append(f"BID:  {best_bid:.4f}")
        else:
            lines.append("BID:  N/A")

        if best_ask is not None:
            lines.append(f"ASK:  {best_ask:.4f}")
        else:
            lines.append("ASK:  N/A")

        if spread is not None:
            lines.append(f"SPREAD: {spread:.4f}")
        else:
            lines.append("SPREAD: N/A")

        return "\n".join(lines)


# Type hint
from typing import Optional

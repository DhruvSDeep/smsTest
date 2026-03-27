"""Market making strategy.

This module provides a market maker strategy that quotes bid and ask prices
around the midprice to provide liquidity to the market.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent
from sim.agents.strategy_interface import Observation, OrderRequest, Strategy


class MarketMakerStrategy(Strategy):
    """Market making strategy that provides liquidity.

    This strategy continuously quotes both bid and ask orders around the
    midprice, profiting from the spread. It manages inventory risk by
    adjusting spreads based on position.

    Attributes:
        spread: Bid-ask spread as a fraction of midprice (e.g., 0.001 = 0.1%)
        order_quantity: Size of each order
        max_position: Maximum position size before reducing activity
        active: Whether the strategy is actively quoting
    """

    def __init__(
        self,
        spread: float = 0.001,
        order_quantity: float = 10.0,
        max_position: float = 100.0,
        seed: Optional[int] = None,
    ):
        """Initialize the market maker strategy.

        Args:
            spread: Bid-ask spread as fraction of midprice
            order_quantity: Size of each order
            max_position: Maximum position before reducing quoting
            seed: Random seed for determinism
        """
        self.spread = spread
        self.order_quantity = order_quantity
        self.max_position = max_position
        self._random = random.Random(seed)
        self._latest_news: Optional[NewsEvent] = None

    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        """Generate market maker orders for all traded commodities.

        Args:
            observations: Dict mapping commodity name -> Observation

        Returns:
            List of order requests (bid and ask per commodity)
        """
        orders = []
        for observation in observations.values():
            # Use reference_price as fallback when midprice is unavailable
            midprice = observation.midprice
            if midprice is None:
                midprice = observation.reference_price
            if midprice is None:
                continue

            # Check if we should be active based on position in this commodity
            if abs(observation.position) >= self.max_position:
                continue

            recent_prices = [price for price, _, _ in observation.last_trades[-5:]]
            trend_bias = 0.0
            if len(recent_prices) >= 2 and midprice > 0:
                trend_bias = (recent_prices[-1] - recent_prices[0]) / midprice

            news_shift = self._latest_news.price_shift(0.008) if self._latest_news else 0.0
            spread_multiplier = (
                self._latest_news.spread_multiplier() if self._latest_news else 1.0
            )
            size_multiplier = (
                self._latest_news.activity_multiplier() if self._latest_news else 1.0
            )

            inventory_bias = 0.003 * (observation.position / max(self.max_position, 1.0))
            reservation_price = midprice * (
                1 + news_shift + 0.35 * trend_bias - inventory_bias
            )
            effective_spread = max(0.0001, self.spread * spread_multiplier)
            quote_step = max(
                midprice * 0.0005,
                (observation.spread or (midprice * effective_spread * 2)) * 0.25,
            )
            bid_price = reservation_price * (1 - effective_spread)
            ask_price = reservation_price * (1 + effective_spread)

            if observation.best_bid is not None:
                bid_price = max(bid_price, observation.best_bid + quote_step)
            if observation.best_ask is not None:
                ask_price = min(ask_price, observation.best_ask - quote_step)
                bid_price = min(bid_price, observation.best_ask - quote_step)
            if observation.best_bid is not None:
                ask_price = max(ask_price, observation.best_bid + quote_step)

            if bid_price >= ask_price:
                midpoint = reservation_price
                bid_price = midpoint - quote_step / 2
                ask_price = midpoint + quote_step / 2

            base_quantity = max(1.0, self.order_quantity * size_multiplier)

            orders.append(
                OrderRequest(
                    side=Side.BID,
                    order_type=OrderType.LIMIT,
                    quantity=base_quantity,
                    commodity=observation.commodity,
                    price=bid_price,
                )
            )
            orders.append(
                OrderRequest(
                    side=Side.ASK,
                    order_type=OrderType.LIMIT,
                    quantity=base_quantity,
                    commodity=observation.commodity,
                    price=ask_price,
                )
            )

        return orders

    def on_news(self, news: Optional[NewsEvent]) -> None:
        """Store the latest news for quote calibration."""
        self._latest_news = news

    def reset(self) -> None:
        """Reset strategy state."""
        self._latest_news = None

    def refresh_orders(self) -> bool:
        """Market makers replace stale quotes every tick."""
        return True

"""Liquidity maker strategy.

This module provides a liquidity maker strategy that provides limit orders
that rest on the book, earning the spread when filled.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent
from sim.agents.strategy_interface import Observation, OrderRequest, Strategy


class LiquidityMakerStrategy(Strategy):
    """Liquidity maker strategy that provides limit orders.

    This strategy places limit orders that rest on the book, earning
    the spread when matched. Unlike market makers, it may only quote
    on one side or adjust prices based on market conditions.

    Attributes:
        order_quantity: Size of each order
        spread: Target spread as fraction of midprice
        probability: Probability of placing an order each tick
        side: Which side to quote (None = both sides)
        active: Whether the strategy is actively providing liquidity
    """

    def __init__(
        self,
        order_quantity: float = 10.0,
        spread: float = 0.002,
        probability: float = 0.5,
        side: Optional[str] = None,  # "bid", "ask", or None for both
        seed: Optional[int] = None,
    ):
        """Initialize the liquidity maker strategy.

        Args:
            order_quantity: Size of each limit order
            spread: Target spread as fraction of midprice
            probability: Probability of placing an order each tick
            side: Which side to quote ("bid", "ask", or None for both)
            seed: Random seed for determinism
        """
        self.order_quantity = order_quantity
        self.spread = spread
        self.probability = probability
        self.side = side
        self._random = random.Random(seed)
        self._latest_news: Optional[NewsEvent] = None

    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        """Generate liquidity maker orders for all traded commodities.

        Args:
            observations: Dict mapping commodity name -> Observation

        Returns:
            List of limit order requests
        """
        orders = []
        for observation in observations.values():
            # Use reference_price as fallback when midprice is unavailable
            midprice = observation.midprice
            if midprice is None:
                midprice = observation.reference_price
            if midprice is None:
                continue

            activity_multiplier = (
                self._latest_news.activity_multiplier() if self._latest_news else 1.0
            )
            adjusted_probability = min(1.0, self.probability * activity_multiplier)

            # Random chance to place order per commodity
            if self._random.random() > adjusted_probability:
                continue

            recent_prices = [price for price, _, _ in observation.last_trades[-4:]]
            trend_bias = 0.0
            if len(recent_prices) >= 2 and midprice > 0:
                trend_bias = (recent_prices[-1] - recent_prices[0]) / midprice
            news_shift = self._latest_news.price_shift(0.006) if self._latest_news else 0.0
            spread_multiplier = (
                self._latest_news.spread_multiplier() if self._latest_news else 1.0
            )
            directional_bias = (
                self._latest_news.directional_bias if self._latest_news else 0.0
            )
            reservation_price = midprice * (1 + news_shift + 0.25 * trend_bias)
            effective_spread = max(0.0001, self.spread * spread_multiplier)
            quote_step = max(
                midprice * 0.0005,
                (observation.spread or (midprice * effective_spread * 2)) * 0.2,
            )

            # Determine which sides to quote
            sides_to_quote = []
            if self.side is None:
                sides_to_quote = [Side.BID, Side.ASK]
            elif self.side.lower() == "bid":
                sides_to_quote = [Side.BID]
            elif self.side.lower() == "ask":
                sides_to_quote = [Side.ASK]

            for side in sides_to_quote:
                if side == Side.BID:
                    price = reservation_price * (1 - effective_spread)
                    if observation.best_bid is not None:
                        price = max(price, observation.best_bid + quote_step)
                    if observation.best_ask is not None:
                        price = min(price, observation.best_ask - quote_step)
                    quantity = max(
                        1.0, self.order_quantity * (1 + max(0.0, directional_bias))
                    )
                else:
                    price = reservation_price * (1 + effective_spread)
                    if observation.best_ask is not None:
                        price = min(price, observation.best_ask - quote_step)
                    if observation.best_bid is not None:
                        price = max(price, observation.best_bid + quote_step)
                    quantity = max(
                        1.0, self.order_quantity * (1 + max(0.0, -directional_bias))
                    )

                orders.append(
                    OrderRequest(
                        side=side,
                        order_type=OrderType.LIMIT,
                        quantity=quantity,
                        commodity=observation.commodity,
                        price=price,
                    )
                )

        return orders

    def on_news(self, news: Optional[NewsEvent]) -> None:
        """Store the latest news for optional quote calibration."""
        self._latest_news = news

    def reset(self) -> None:
        """Reset strategy state."""
        self._latest_news = None

    def refresh_orders(self) -> bool:
        """Liquidity makers refresh passive quotes each tick."""
        return True

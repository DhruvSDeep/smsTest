"""Random trader strategy.

This module provides a simple random trader strategy that places
random orders for testing and simulation purposes.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent
from sim.agents.strategy_interface import Observation, OrderRequest, Strategy


class RandomTraderStrategy(Strategy):
    """Simple random trader for testing and simulation.

    Places random orders with configurable probability and size.
    Useful for creating market activity and testing exchange functionality.

    Attributes:
        order_quantity: Size of each order
        probability: Probability of placing an order each tick
        active: Whether the strategy is actively trading
    """

    def __init__(
        self,
        order_quantity: float = 10.0,
        probability: float = 0.3,
        seed: Optional[int] = None,
    ):
        """Initialize the random trader strategy.

        Args:
            order_quantity: Size of each order
            probability: Probability of placing an order each tick
            seed: Random seed for determinism
        """
        self.order_quantity = order_quantity
        self.probability = probability
        self._random = random.Random(seed)
        self._latest_news: Optional[NewsEvent] = None

    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        """Generate random orders across all traded commodities.

        Args:
            observations: Dict mapping commodity name -> Observation

        Returns:
            List of random order requests
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

            # Random side
            side = self._random.choice([Side.BID, Side.ASK])
            if self._latest_news is not None and self._random.random() < min(
                1.0, abs(self._latest_news.directional_bias)
            ):
                side = Side.BID if self._latest_news.directional_bias >= 0 else Side.ASK

            quantity = max(1.0, self.order_quantity * activity_multiplier)

            if side == Side.BID and observation.cash < quantity * midprice:
                continue

            # Occasionally cross the spread to ensure this trader actually interacts.
            if (
                observation.best_bid is not None
                and observation.best_ask is not None
                and self._random.random() < 0.35
            ):
                orders.append(
                    OrderRequest(
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=quantity,
                        commodity=observation.commodity,
                        price=None,
                    )
                )
                continue

            # Random price around midprice (within spread)
            if observation.spread and observation.spread > 0:
                offset = self._random.uniform(-observation.spread, observation.spread)
            else:
                offset = self._random.uniform(-0.1, 0.1)

            news_shift = self._latest_news.price_shift(0.01) if self._latest_news else 0.0

            if side == Side.BID:
                price = midprice + offset
                if observation.best_bid is not None:
                    price = max(price, observation.best_bid + max(midprice * 0.0003, 0.01))
                if observation.best_ask is not None:
                    price = min(price, observation.best_ask - max(midprice * 0.0003, 0.01))
                price = min(price, midprice * (1.002 + max(0.0, news_shift)))
            else:
                price = midprice - offset
                if observation.best_ask is not None:
                    price = min(price, observation.best_ask - max(midprice * 0.0003, 0.01))
                if observation.best_bid is not None:
                    price = max(price, observation.best_bid + max(midprice * 0.0003, 0.01))
                price = max(price, midprice * (0.998 + max(0.0, -news_shift)))

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
        """Store the latest news for optional calibration."""
        self._latest_news = news

    def reset(self) -> None:
        """Reset strategy state."""
        self._latest_news = None

    def refresh_orders(self) -> bool:
        """Random traders should not leave stale passive orders behind."""
        return True

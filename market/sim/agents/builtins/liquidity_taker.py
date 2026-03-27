"""Liquidity taker strategy.

This module provides a liquidity taker strategy that takes liquidity
from the market by placing market orders that cross the spread.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent
from sim.agents.strategy_interface import Observation, OrderRequest, Strategy


class LiquidityTakerStrategy(Strategy):
    """Liquidity taker strategy that crosses the spread.

    This strategy places market orders to immediately fill orders,
    taking liquidity from the order book. It prioritizes execution
    over getting a better price.

    Attributes:
        order_quantity: Size of each order
        probability: Probability of placing an order each tick
        active: Whether the strategy is actively taking liquidity
    """

    def __init__(
        self,
        order_quantity: float = 10.0,
        probability: float = 0.3,
        seed: Optional[int] = None,
    ):
        """Initialize the liquidity taker strategy.

        Args:
            order_quantity: Size of each market order
            probability: Probability of placing an order each tick
            seed: Random seed for determinism
        """
        self.order_quantity = order_quantity
        self.probability = probability
        self._random = random.Random(seed)
        self._latest_news: Optional[NewsEvent] = None

    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        """Generate liquidity taker orders across all traded commodities.

        Args:
            observations: Dict mapping commodity name -> Observation

        Returns:
            List of market order requests
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

            quantity_multiplier = (
                self._latest_news.activity_multiplier() if self._latest_news else 1.0
            )
            quantity = max(1.0, self.order_quantity * quantity_multiplier)

            inventory_limit = max(quantity * 3, self.order_quantity * 2)
            if observation.position >= inventory_limit:
                side = Side.ASK
            elif observation.position <= -inventory_limit:
                side = Side.BID
            else:
                recent_prices = [price for price, _, _ in observation.last_trades[-4:]]
                trend_bias = 0.0
                if len(recent_prices) >= 2 and midprice > 0:
                    trend_bias = (recent_prices[-1] - recent_prices[0]) / midprice

                directional_bias = (
                    self._latest_news.directional_bias if self._latest_news else 0.0
                )
                combined_bias = trend_bias + 0.25 * directional_bias

                if self._random.random() < 0.35:
                    side = self._random.choice([Side.BID, Side.ASK])
                else:
                    side = Side.BID if combined_bias >= 0 else Side.ASK

            if self._latest_news is not None and self._random.random() < min(
                1.0, abs(self._latest_news.directional_bias)
            ):
                side = Side.BID if self._latest_news.directional_bias >= 0 else Side.ASK

            if side == Side.BID and observation.cash < quantity * midprice:
                continue

            orders.append(
                OrderRequest(
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=quantity,
                    commodity=observation.commodity,
                    price=None,  # Market orders don't need a price
                )
            )
        return orders

    def on_news(self, news: Optional[NewsEvent]) -> None:
        """Store the latest news for directional calibration."""
        self._latest_news = news

    def reset(self) -> None:
        """Reset strategy state."""
        self._latest_news = None

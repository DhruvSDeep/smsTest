"""Strategy interface for trading agents.

This module defines the contract that all trading strategies must follow.
Strategies receive observations and return orders - they do not have
direct access to the exchange.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent


@dataclass
class Observation:
    """Observation snapshot passed to strategies.

    This is the only input a strategy receives - it provides a complete
    view of the current market state without direct exchange access.

    Attributes:
        tick: Current simulation tick
        best_bid: Best bid price
        best_ask: Best ask price
        midprice: Midprice (average of best bid/ask)
        spread: Bid-ask spread
        reference_price: Fallback price when midprice is unavailable (e.g., from fundamentals)
        last_trades: List of recent trades (price, quantity, tick)
        position: Agent's current position (positive = long)
        cash: Agent's current cash balance
        bid_depth: List of (price, quantity) tuples for bids
        ask_depth: List of (price, quantity) tuples for asks
        news: Most recently received structured news event
    """

    tick: float
    commodity: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    midprice: Optional[float]
    spread: Optional[float]
    reference_price: Optional[float]  # Fallback price from fundamentals
    last_trades: List[tuple[float, float, float]]  # (price, quantity, tick)
    position: float
    cash: float
    bid_depth: List[tuple[float, float]]
    ask_depth: List[tuple[float, float]]
    news: Optional[NewsEvent] = None


@dataclass
class OrderRequest:
    """Request to place an order.

    Attributes:
        side: BID or ASK
        order_type: LIMIT or MARKET
        quantity: Order quantity
        price: Price for limit orders (None for market)
    """

    side: Side
    order_type: OrderType
    quantity: float
    commodity: str = ""
    price: Optional[float] = None


class Strategy(ABC):
    """Abstract base class for trading strategies.

    All strategies must implement the `act` method which takes an observation
    and returns a list of order requests.

    Important:
        - Strategies have NO direct access to the exchange
        - Strategies have NO access to global mutable state
        - Strategies must be deterministic given the same observation
        - Strategies should be stateless (any state should be in the agent)
    """

    @abstractmethod
    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        """Generate orders based on observations across all traded commodities.

        Args:
            observations: Dict mapping commodity name -> Observation

        Returns:
            List of order requests to submit
        """
        pass

    def reset(self) -> None:
        """Reset strategy state for a new simulation run.

        Override this if your strategy maintains internal state
        that needs to be reset between runs.
        """
        pass

    def on_news(self, news: Optional[NewsEvent]) -> None:
        """Receive structured news from the main simulation loop.

        Strategies may use this callback to calibrate internal parameters,
        cache the latest event, or ignore it entirely.
        """
        pass

    def refresh_orders(self) -> bool:
        """Whether previously resting orders should be canceled before acting."""
        return False


class EmptyStrategy(Strategy):
    """Default strategy that does nothing.

    Useful as a placeholder or for testing.
    """

    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        """Return no orders."""
        return []


class RandomStrategy(Strategy):
    """Simple random strategy for testing.

    Places random orders with no real logic.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional seed for determinism."""
        import random

        self._random = random.Random(seed)
        self._latest_news: Optional[NewsEvent] = None

    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        """Generate random orders across all observed commodities."""
        orders = []
        for observation in observations.values():
            # Use reference_price as fallback when midprice is unavailable
            midprice = observation.midprice
            if midprice is None:
                midprice = observation.reference_price
            if midprice is None:
                continue

            # 20% chance to place an order each tick per commodity
            if self._random.random() > 0.2:
                continue

            # Random side
            side = self._random.choice([Side.BID, Side.ASK])
            if self._latest_news is not None and self._random.random() < abs(
                self._latest_news.directional_bias
            ):
                side = Side.BID if self._latest_news.directional_bias >= 0 else Side.ASK

            # Random quantity
            quantity = self._random.uniform(1, 10)

            # Random price around midprice
            if observation.spread:
                offset = self._random.uniform(-observation.spread, observation.spread)
            else:
                offset = self._random.uniform(-0.1, 0.1)

            price = midprice + offset if side == Side.BID else midprice - offset

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
        """Store the latest news for optional use in `act`."""
        self._latest_news = news

    def reset(self) -> None:
        """Reset strategy state."""
        self._latest_news = None

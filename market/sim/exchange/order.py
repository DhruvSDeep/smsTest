"""Order representation for the exchange.

This module defines the core order types and order data structures
used throughout the market simulation.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Optional


class OrderType(enum.Enum):
    """Type of order."""

    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


class Side(enum.Enum):
    """Side of the market."""

    BID = "bid"
    ASK = "ask"


@total_ordering
@dataclass
class Order:
    """Represents a trading order.

    Orders are ordered by price-time priority:
    - For bids: higher prices have priority (descending)
    - For asks: lower prices have priority (ascending)
    - For same price: earlier timestamp has priority

    Attributes:
        order_id: Unique identifier for the order
        agent_id: ID of the agent that placed the order
        side: BID or ASK
        order_type: LIMIT, MARKET, or CANCEL
        price: Price for limit orders (None for market orders)
        quantity: Quantity to trade
        timestamp: Order creation time (for priority)
        filled_quantity: Amount already filled
        canceled: Whether the order has been canceled

    Note:
        The order=True in dataclass makes __eq__ and __hash__ based on
        all fields, but we override __lt__ for price-time priority ordering.
        This is needed because dataclass with order=True doesn't work well
        with custom __lt__ that depends on side.
    """

    order_id: int
    agent_id: int
    side: Side
    order_type: OrderType
    price: Optional[float]
    quantity: float
    commodity: str
    timestamp: float = field(compare=True)
    filled_quantity: float = field(default=0.0, compare=False)
    canceled: bool = field(default=False, compare=False)
    original_quantity: float = field(default=0.0, compare=False)

    def __post_init__(self) -> None:
        """Initialize derived fields after initialization."""
        if self.original_quantity == 0.0:
            self.original_quantity = self.quantity

    def __lt__(self, other: Order) -> bool:
        """Compare orders for priority queue ordering.

        For bids: higher price = higher priority
        For asks: lower price = higher priority
        For same price: earlier timestamp = higher priority
        """
        if self.price is None and other.price is None:
            return self.timestamp < other.timestamp
        if self.price is None:
            return False  # Market orders have lowest priority
        if other.price is None:
            return True

        if self.side == other.side:
            if self.side == Side.BID:
                # Higher bid price = higher priority
                if self.price != other.price:
                    return self.price > other.price
            else:
                # Lower ask price = higher priority
                if self.price != other.price:
                    return self.price < other.price

            # Same price: earlier timestamp = higher priority
            return self.timestamp < other.timestamp

        # Different sides - shouldn't happen in same book
        return self.timestamp < other.timestamp

    @property
    def remaining_quantity(self) -> float:
        """Returns the remaining quantity to be filled."""
        return self.quantity - self.filled_quantity

    @property
    def is_filled(self) -> bool:
        """Returns True if the order is fully filled."""
        return self.filled_quantity >= self.quantity

    @property
    def is_market(self) -> bool:
        """Returns True if this is a market order."""
        return self.order_type == OrderType.MARKET

    @property
    def is_limit(self) -> bool:
        """Returns True if this is a limit order."""
        return self.order_type == OrderType.LIMIT

    def cancel(self) -> None:
        """Mark the order as canceled."""
        self.canceled = True

    def fill(self, quantity: float) -> float:
        """Fill part of the order.

        Args:
            quantity: The quantity to fill

        Returns:
            The actual quantity filled (can be less if order doesn't have enough remaining)
        """
        remaining = self.remaining_quantity
        filled = min(quantity, remaining)
        self.filled_quantity += filled
        return filled


@dataclass
class Trade:
    """Represents a trade execution.

    Attributes:
        trade_id: Unique identifier for the trade
        maker_order_id: ID of the maker order
        taker_order_id: ID of the taker order
        maker_agent_id: ID of the maker agent
        taker_agent_id: ID of the taker agent
        price: Trade execution price
        quantity: Trade quantity
        timestamp: Trade execution time
    """

    trade_id: int
    maker_order_id: int
    taker_order_id: int
    maker_agent_id: int
    taker_agent_id: int
    price: float
    quantity: float
    timestamp: float
    commodity: str = ""
    taker_side: Optional[Side] = None

    def __repr__(self) -> str:
        return (
            f"Trade(id={self.trade_id}, maker={self.maker_agent_id}, "
            f"taker={self.taker_agent_id}, price={self.price:.4f}, "
            f"qty={self.quantity:.4f})"
        )


@dataclass
class OrderStatus:
    """Status of an order after processing.

    Attributes:
        order_id: The order this status relates to
        filled_quantity: How much was filled
        remaining_quantity: How much is still remaining
        canceled: Whether the order was canceled
        trades: List of trades resulting from this order
    """

    order_id: int
    filled_quantity: float
    remaining_quantity: float
    canceled: bool
    trades: list[Trade] = field(default_factory=list)

    @property
    def is_filled(self) -> bool:
        return self.remaining_quantity <= 0

    @property
    def is_partially_filled(self) -> bool:
        return self.filled_quantity > 0 and not self.is_filled

"""Order book implementation for the exchange.

This module provides the order book data structure with efficient
price-time priority matching logic.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .order import Order, OrderType, Side, Trade


class OrderBook:
    """Order book for a single trading instrument.

    The order book maintains separate heaps for bid (buy) and ask (sell) orders,
    with price-time priority ordering:
    - Bids: highest price has priority
    - Asks: lowest price has priority
    - Same price: earliest timestamp has priority

    Attributes:
        best_bid: The highest bid price (None if no bids)
        best_ask: The lowest ask price (None if no asks)
    """

    def __init__(self) -> None:
        """Initialize an empty order book."""
        # Priority queues (heaps) for orders
        # Use negative price for bids (to get max-heap behavior)
        self._bid_heap: list[Order] = []  # max-heap via negative price
        self._ask_heap: list[Order] = []  # min-heap

        # Order lookup for efficient removal
        self._orders_by_id: Dict[int, Order] = {}
        self._agent_orders: Dict[int, list[int]] = defaultdict(list)

        # Track best bid/ask
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None

    @property
    def best_bid(self) -> Optional[float]:
        """Returns the best bid price."""
        self._clean_heap(Side.BID)
        if self._bid_heap:
            return self._bid_heap[0].price
        return None

    @property
    def best_ask(self) -> Optional[float]:
        """Returns the best ask price."""
        self._clean_heap(Side.ASK)
        if self._ask_heap:
            return self._ask_heap[0].price
        return None

    @property
    def spread(self) -> Optional[float]:
        """Returns the bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def midprice(self) -> Optional[float]:
        """Returns the midprice (average of best bid and ask)."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    def add_order(self, order: Order) -> None:
        """Add an order to the book.

        Args:
            order: The order to add

        Raises:
            ValueError: If the order is already in the book
        """
        if order.order_id in self._orders_by_id:
            raise ValueError(f"Order {order.order_id} already in book")

        if order.order_type == OrderType.MARKET:
            # Market orders go to a special queue for immediate matching
            pass

        self._orders_by_id[order.order_id] = order
        self._agent_orders[order.agent_id].append(order.order_id)

        # Add to appropriate heap
        # For bids: use negative price for max-heap simulation
        # For asks: use positive price for min-heap
        if order.side == Side.BID:
            heapq.heappush(self._bid_heap, order)
            # Update best bid
            if self._best_bid is None or order.price > self._best_bid:
                self._best_bid = order.price
        else:
            heapq.heappush(self._ask_heap, order)
            # Update best ask
            if self._best_ask is None or order.price < self._best_ask:
                self._best_ask = order.price

    def remove_order(self, order_id: int) -> Optional[Order]:
        """Remove an order from the book.

        Args:
            order_id: The ID of the order to remove

        Returns:
            The removed order, or None if not found
        """
        order = self._orders_by_id.pop(order_id, None)
        if order is not None:
            order.cancel()
            # Remove from agent's order list
            if order.agent_id in self._agent_orders:
                try:
                    self._agent_orders[order.agent_id].remove(order_id)
                except ValueError:
                    pass
        return order

    def cancel_order(self, order_id: int) -> Optional[Order]:
        """Cancel an order (mark as canceled but keep in heaps).

        This is more efficient than remove_order for matching purposes.

        Args:
            order_id: The ID of the order to cancel

        Returns:
            The canceled order, or None if not found
        """
        order = self._orders_by_id.get(order_id)
        if order is not None:
            order.cancel()
        return order

    def get_order(self, order_id: int) -> Optional[Order]:
        """Get an order by ID."""
        return self._orders_by_id.get(order_id)

    def get_best_bid(self) -> Optional[Order]:
        """Get the best bid order (highest priority)."""
        self._clean_heap(Side.BID)
        if self._bid_heap:
            return self._bid_heap[0]
        return None

    def get_best_ask(self) -> Optional[Order]:
        """Get the best ask order (lowest priority)."""
        self._clean_heap(Side.ASK)
        if self._ask_heap:
            return self._ask_heap[0]
        return None

    def _clean_heap(self, side: Side) -> None:
        """Remove canceled/filled orders from heap top."""
        if side == Side.BID:
            while self._bid_heap and (
                self._bid_heap[0].canceled or self._bid_heap[0].is_filled
            ):
                heapq.heappop(self._bid_heap)
        else:
            while self._ask_heap and (
                self._ask_heap[0].canceled or self._ask_heap[0].is_filled
            ):
                heapq.heappop(self._ask_heap)

    def get_orders_at_price(self, side: Side, price: float) -> List[Order]:
        """Get all orders at a specific price level.

        Args:
            side: BID or ASK
            price: The price level

        Returns:
            List of orders at that price (ordered by time priority)
        """
        # This is O(n) - could be optimized with a price-level structure
        if side == Side.BID:
            heap = self._bid_heap
        else:
            heap = self._ask_heap

        result = []
        for order in heap:
            if not order.canceled and not order.is_filled:
                if order.price == price:
                    result.append(order)

        # Sort by timestamp for time priority
        result.sort(key=lambda o: o.timestamp)
        return result

    def get_level2_snapshot(
        self, depth: int = 10
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Get a level 2 market snapshot.

        Args:
            depth: Number of price levels to include

        Returns:
            Dictionary with 'bids' and 'asks', each a list of (price, quantity) tuples
        """
        bids = []
        asks = []

        # Collect bid levels (need to filter canceled/filled)
        bid_prices = defaultdict(float)
        for order in self._bid_heap:
            if not order.canceled and not order.is_filled:
                if order.price is not None:
                    bid_prices[order.price] += order.remaining_quantity

        # Sort by price (descending for bids) and take top N
        sorted_bids = sorted(bid_prices.items(), reverse=True)[:depth]
        bids = sorted_bids

        # Collect ask levels
        ask_prices = defaultdict(float)
        for order in self._ask_heap:
            if not order.canceled and not order.is_filled:
                if order.price is not None:
                    ask_prices[order.price] += order.remaining_quantity

        # Sort by price (ascending for asks) and take top N
        sorted_asks = sorted(ask_prices.items())[:depth]
        asks = sorted_asks

        return {"bids": bids, "asks": asks}

    def get_total_volume(self, side: Optional[Side] = None) -> float:
        """Get total volume in the book.

        Args:
            side: Optional side to filter by

        Returns:
            Total quantity available
        """
        total = 0.0

        if side is None or side == Side.BID:
            for order in self._bid_heap:
                if not order.canceled and not order.is_filled:
                    total += order.remaining_quantity

        if side is None or side == Side.ASK:
            for order in self._ask_heap:
                if not order.canceled and not order.is_filled:
                    total += order.remaining_quantity

        return total

    def get_volume_at_price(self, side: Side, price: float) -> float:
        """Get total volume at a specific price level."""
        total = 0.0
        heap = self._bid_heap if side == Side.BID else self._ask_heap

        for order in heap:
            if not order.canceled and not order.is_filled and order.price == price:
                total += order.remaining_quantity

        return total

    def get_orders_for_agent(self, agent_id: int) -> List[Order]:
        """Get all active orders for an agent."""
        order_ids = self._agent_orders.get(agent_id, [])
        result = []
        for order_id in order_ids:
            order = self._orders_by_id.get(order_id)
            if order and not order.canceled and not order.is_filled:
                result.append(order)
        return result

    def clear(self) -> None:
        """Clear all orders from the book."""
        self._bid_heap.clear()
        self._ask_heap.clear()
        self._orders_by_id.clear()
        self._agent_orders.clear()
        self._best_bid = None
        self._best_ask = None

    def __len__(self) -> int:
        """Return number of active orders in the book."""
        count = 0
        for order in self._orders_by_id.values():
            if not order.canceled and not order.is_filled:
                count += 1
        return count

    def __repr__(self) -> str:
        return (
            f"OrderBook(best_bid={self.best_bid}, best_ask={self.best_ask}, "
            f"spread={self.spread}, orders={len(self)})"
        )

"""Matching engine for the exchange.

This module implements the core matching logic that processes orders
and generates trades according to price-time priority.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .order import Order, OrderStatus, OrderType, Side, Trade
from .orderbook import OrderBook


class MatchingEngine:
    """Matching engine supporting multiple commodity order books.

    Each commodity has its own independent order book. Orders carry a
    commodity field that routes them to the correct book.

    Attributes:
        order_books: Dict mapping commodity name -> OrderBook
        commodities: Ordered list of commodity names
        tick: Current tick counter
    """

    def __init__(self, commodities: List[str]) -> None:
        """Initialize the matching engine.

        Args:
            commodities: List of commodity names to create books for
        """
        self.commodities: List[str] = list(commodities)
        self.order_books: Dict[str, OrderBook] = {c: OrderBook() for c in commodities}
        self.tick: float = 0.0
        self._trade_counter: int = 0
        self._order_counter: int = 0
        self._last_trades: Dict[str, List[Trade]] = {c: [] for c in commodities}

    def reset(self) -> None:
        """Reset the matching engine to initial state."""
        for ob in self.order_books.values():
            ob.clear()
        self.tick = 0.0
        self._trade_counter = 0
        self._order_counter = 0
        for key in self._last_trades:
            self._last_trades[key].clear()

    def next_tick(self) -> None:
        """Advance to the next tick."""
        self.tick += 1
        for key in self._last_trades:
            self._last_trades[key].clear()

    def get_order_book(self, commodity: str) -> OrderBook:
        """Get the order book for a commodity.

        Args:
            commodity: Commodity name

        Returns:
            OrderBook for the commodity

        Raises:
            KeyError: If commodity is not registered
        """
        return self.order_books[commodity]

    def create_order(
        self,
        agent_id: int,
        side: Side,
        order_type: OrderType,
        quantity: float,
        commodity: str,
        price: Optional[float] = None,
    ) -> Order:
        """Create a new order with a unique ID.

        Args:
            agent_id: The agent placing the order
            side: BID or ASK
            order_type: LIMIT, MARKET, or CANCEL
            quantity: Order quantity
            commodity: Commodity to trade
            price: Price for limit orders (None for market orders)

        Returns:
            The created order
        """
        self._order_counter += 1
        return Order(
            order_id=self._order_counter,
            agent_id=agent_id,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            commodity=commodity,
            timestamp=self.tick,
        )

    def submit_order(self, order: Order) -> OrderStatus:
        """Submit an order for matching.

        Routes the order to the correct commodity book via order.commodity.

        Args:
            order: The order to submit

        Returns:
            OrderStatus with fill information
        """
        if order.order_type == OrderType.CANCEL:
            return self._cancel_order(order)

        if order.order_type == OrderType.MARKET:
            return self._match_market_order(order)

        return self._match_limit_order(order)

    def _cancel_order(self, order: Order) -> OrderStatus:
        """Cancel an existing order."""
        ob = self.order_books[order.commodity]
        existing = ob.remove_order(order.order_id)
        if existing:
            return OrderStatus(
                order_id=order.order_id,
                filled_quantity=existing.filled_quantity,
                remaining_quantity=existing.remaining_quantity,
                canceled=True,
            )
        return OrderStatus(
            order_id=order.order_id,
            filled_quantity=0.0,
            remaining_quantity=0.0,
            canceled=True,
        )

    def _match_limit_order(self, order: Order) -> OrderStatus:
        """Match a limit order against the commodity's book."""
        ob = self.order_books[order.commodity]
        trades: List[Trade] = []
        filled_quantity = 0.0

        if order.side == Side.BID:
            while order.remaining_quantity > 0:
                best_ask = ob.get_best_ask()
                if best_ask is None:
                    break
                if order.price is not None and order.price < best_ask.price:
                    break

                trade_qty = min(order.remaining_quantity, best_ask.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_ask,
                    taker_order=order,
                    price=best_ask.price,
                    quantity=trade_qty,
                    commodity=order.commodity,
                )
                trades.append(trade)
                order.fill(trade_qty)
                best_ask.fill(trade_qty)
                filled_quantity += trade_qty

                if best_ask.is_filled:
                    ob.remove_order(best_ask.order_id)

        else:
            while order.remaining_quantity > 0:
                best_bid = ob.get_best_bid()
                if best_bid is None:
                    break
                if order.price is not None and order.price > best_bid.price:
                    break

                trade_qty = min(order.remaining_quantity, best_bid.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_bid,
                    taker_order=order,
                    price=best_bid.price,
                    quantity=trade_qty,
                    commodity=order.commodity,
                )
                trades.append(trade)
                order.fill(trade_qty)
                best_bid.fill(trade_qty)
                filled_quantity += trade_qty

                if best_bid.is_filled:
                    ob.remove_order(best_bid.order_id)

        if not order.is_filled:
            ob.add_order(order)

        self._last_trades[order.commodity].extend(trades)

        return OrderStatus(
            order_id=order.order_id,
            filled_quantity=filled_quantity,
            remaining_quantity=order.remaining_quantity,
            canceled=False,
            trades=trades,
        )

    def _match_market_order(self, order: Order) -> OrderStatus:
        """Match a market order against the commodity's book."""
        ob = self.order_books[order.commodity]
        trades: List[Trade] = []
        filled_quantity = 0.0

        if order.side == Side.BID:
            while order.remaining_quantity > 0:
                best_ask = ob.get_best_ask()
                if best_ask is None:
                    break

                trade_qty = min(order.remaining_quantity, best_ask.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_ask,
                    taker_order=order,
                    price=best_ask.price,
                    quantity=trade_qty,
                    commodity=order.commodity,
                )
                trades.append(trade)
                order.fill(trade_qty)
                best_ask.fill(trade_qty)
                filled_quantity += trade_qty

                if best_ask.is_filled:
                    ob.remove_order(best_ask.order_id)

        else:
            while order.remaining_quantity > 0:
                best_bid = ob.get_best_bid()
                if best_bid is None:
                    break

                trade_qty = min(order.remaining_quantity, best_bid.remaining_quantity)
                trade = self._create_trade(
                    maker_order=best_bid,
                    taker_order=order,
                    price=best_bid.price,
                    quantity=trade_qty,
                    commodity=order.commodity,
                )
                trades.append(trade)
                order.fill(trade_qty)
                best_bid.fill(trade_qty)
                filled_quantity += trade_qty

                if best_bid.is_filled:
                    ob.remove_order(best_bid.order_id)

        self._last_trades[order.commodity].extend(trades)

        return OrderStatus(
            order_id=order.order_id,
            filled_quantity=filled_quantity,
            remaining_quantity=order.remaining_quantity,
            canceled=False,
            trades=trades,
        )

    def _create_trade(
        self,
        maker_order: Order,
        taker_order: Order,
        price: float,
        quantity: float,
        commodity: str,
    ) -> Trade:
        """Create a trade record."""
        self._trade_counter += 1
        return Trade(
            trade_id=self._trade_counter,
            maker_order_id=maker_order.order_id,
            taker_order_id=taker_order.order_id,
            maker_agent_id=maker_order.agent_id,
            taker_agent_id=taker_order.agent_id,
            price=price,
            quantity=quantity,
            timestamp=self.tick,
            commodity=commodity,
            taker_side=taker_order.side,
        )

    def batch_match(self, orders: List[Order]) -> Dict[int, OrderStatus]:
        """Process a batch of orders.

        Args:
            orders: List of orders to process

        Returns:
            Dictionary mapping order_id to OrderStatus
        """
        results: Dict[int, OrderStatus] = {}
        for order in orders:
            status = self.submit_order(order)
            results[order.order_id] = status
        return results

    def get_last_trades(self, commodity: str) -> List[Trade]:
        """Get the trades from the last matching round for a commodity."""
        return self._last_trades[commodity].copy()

    def get_market_state(self, commodity: str) -> Dict:
        """Get current market state snapshot for a commodity.

        Args:
            commodity: Commodity name

        Returns:
            Dictionary with market state information
        """
        ob = self.order_books[commodity]
        return {
            "tick": self.tick,
            "commodity": commodity,
            "best_bid": ob.best_bid,
            "best_ask": ob.best_ask,
            "midprice": ob.midprice,
            "spread": ob.spread,
            "volume": ob.get_total_volume(),
            "bid_volume": ob.get_total_volume(Side.BID),
            "ask_volume": ob.get_total_volume(Side.ASK),
            "order_count": len(ob),
            "last_trades": self._last_trades[commodity].copy(),
        }

    def get_depth_snapshot(self, commodity: str, depth: int = 10) -> Dict:
        """Get order book depth snapshot for a commodity.

        Args:
            commodity: Commodity name
            depth: Number of price levels

        Returns:
            Dictionary with bids, asks, best_bid, best_ask
        """
        ob = self.order_books[commodity]
        level2 = ob.get_level2_snapshot(depth)
        return {
            "bids": level2["bids"],
            "asks": level2["asks"],
            "best_bid": ob.best_bid,
            "best_ask": ob.best_ask,
        }

    def __repr__(self) -> str:
        parts = ", ".join(
            f"{c}(bid={self.order_books[c].best_bid}, ask={self.order_books[c].best_ask})"
            for c in self.commodities
        )
        return f"MatchingEngine(tick={self.tick}, [{parts}])"

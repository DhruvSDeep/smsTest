"""Mesa agent wrapper for trading agents.

This module provides the TradingAgent class that combines Mesa's Agent
with our strategy system, portfolio management, and trade handling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from mesa import Agent as MesaAgent

from ..exchange import Order, OrderStatus, OrderType, Side, Trade
from ..news import NewsEvent

if TYPE_CHECKING:
    from ..model import MarketModel


class TradingAgent(MesaAgent):
    """Trading agent that combines Mesa Agent with trading strategy.

    Responsibilities:
    - Hold portfolio state (cash, position)
    - Hold strategy instance
    - Call strategy each tick to generate orders
    - Receive fills from matching engine
    - Update PnL / inventory

    Attributes:
        cash: Current cash balance
        position: Current position (positive = long)
        initial_cash: Starting cash for PnL calculation
        strategy: The trading strategy instance
    """

    def __init__(
        self,
        model: MarketModel,
        strategy: "Optional[Strategy]" = None,
        initial_cash: float = 10000.0,
        commodities: "Optional[List[str]]" = None,
        **kwargs,
    ):
        """Initialize the trading agent.

        Args:
            model: The market model
            strategy: Trading strategy instance (optional)
            initial_cash: Starting cash balance
            commodities: List of commodities this agent trades
            **kwargs: Additional arguments for Mesa Agent
        """
        super().__init__(model, **kwargs)

        # Commodity assignment
        self.commodities: List[str] = list(commodities or [])

        # Portfolio state
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.positions: Dict[str, float] = {c: 0.0 for c in self.commodities}

        # Strategy
        self.strategy = strategy

        # Track orders and trades
        self._pending_orders: Dict[int, Order] = {}
        self._filled_trades: List[Trade] = []

        # PnL tracking
        self._realized_pnl: float = 0.0
        self._trade_history: List[Dict] = []
        self.current_news: Optional[NewsEvent] = None
        self._news_history: List[NewsEvent] = []

    def _mark_price(self, commodity: str) -> float:
        """Return the best available price for marking a commodity position."""
        model = self.model
        midprice = model.exchange.get_order_book(commodity).midprice
        if midprice is not None:
            return midprice

        env = model.environments[commodity]
        env_state = env.get_state()
        return (
            env_state.get("fundamental")
            or env_state.get("current_price")
            or env.initial_price
        )

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized PnL based on current positions and midprices."""
        return 0.0  # TODO: Implement properly with avg cost basis per commodity

    @property
    def total_pnl(self) -> float:
        """Total PnL (realized + unrealized)."""
        return self.equity - self.initial_cash

    @property
    def equity(self) -> float:
        """Total equity (cash + value of all positions)."""
        return self.cash + sum(
            qty * self._mark_price(c) for c, qty in self.positions.items()
        )

    @property
    def return_pct(self) -> float:
        """Return percentage since start."""
        if self.initial_cash == 0:
            return 0.0
        return ((self.equity - self.initial_cash) / self.initial_cash) * 100

    def set_strategy(self, strategy: "Strategy") -> None:
        """Set or swap the trading strategy.

        Args:
            strategy: New strategy instance
        """
        # Reset old strategy if exists
        if self.strategy is not None:
            self.strategy.reset()

        self.strategy = strategy

    def get_observations(self) -> "Dict[str, Observation]":
        """Build observation snapshots for all traded commodities.

        Returns:
            Dict mapping commodity name -> Observation
        """
        from .strategy_interface import Observation

        model: MarketModel = self.model
        exchange = model.exchange
        observations = {}

        for commodity in self.commodities:
            market_state = exchange.get_market_state(commodity)
            depth = exchange.get_depth_snapshot(commodity)
            env_state = model.environments[commodity].get_state()
            reference_price = env_state.get("fundamental") or env_state.get("current_price")

            last_trades = [
                (t.price, t.quantity, t.timestamp)
                for t in market_state.get("last_trades", [])
            ]

            observations[commodity] = Observation(
                tick=market_state["tick"],
                commodity=commodity,
                best_bid=market_state["best_bid"],
                best_ask=market_state["best_ask"],
                midprice=market_state["midprice"],
                spread=market_state["spread"],
                reference_price=reference_price,
                last_trades=last_trades,
                position=self.positions.get(commodity, 0.0),
                cash=self.cash,
                bid_depth=depth["bids"],
                ask_depth=depth["asks"],
                news=self.current_news,
            )

        return observations

    def receive_news(self, news: Optional[NewsEvent]) -> None:
        """Receive a structured news event from the main loop."""
        self.current_news = news
        if news is not None:
            self._news_history.append(news)
        if self.strategy is not None:
            self.strategy.on_news(news)

    def step(self) -> None:
        """Execute one step of the agent's decision making.

        This is called by the Mesa scheduler each tick.
        """
        if (
            self.strategy is not None
            and self.strategy.refresh_orders()
            and self._pending_orders
        ):
            self.cancel_all_orders()

        # Get observations after optional quote refresh.
        observations = self.get_observations()

        # Ask strategy for orders
        if self.strategy is not None:
            order_requests = self.strategy.act(observations)
        else:
            order_requests = []

        # Convert requests to orders and submit
        for request in order_requests:
            self._submit_order(request)

    def _submit_order(self, request: "OrderRequest") -> OrderStatus:
        """Submit an order request to the exchange.

        Args:
            request: The order request

        Returns:
            OrderStatus from the matching engine
        """
        model: MarketModel = self.model

        # Create order
        order = model.exchange.create_order(
            agent_id=self.unique_id,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            commodity=request.commodity or self.commodity,
            price=request.price,
        )

        # Track the order
        self._pending_orders[order.order_id] = order

        # Submit to matching engine
        status = model.exchange.submit_order(order)

        # Process fills for both sides of each trade.
        self._process_fills(order, status)

        # Market orders never rest on the book, so they should not remain pending.
        if order.order_type == OrderType.MARKET:
            self._pending_orders.pop(order.order_id, None)

        return status

    def _process_fills(self, order: Order, status: OrderStatus) -> None:
        """Process order fill results.

        Args:
            order: The order that was processed
            status: The status returned from matching
        """
        for trade in status.trades:
            self._record_trade_fill(trade, order.side, role="taker")

            maker_agent = self.model.get_agent(trade.maker_agent_id)
            if maker_agent is not None and maker_agent.unique_id != self.unique_id:
                maker_side = Side.ASK if order.side == Side.BID else Side.BID
                maker_agent._record_trade_fill(trade, maker_side, role="maker")

        # Remove from pending orders if fully filled or canceled
        if status.is_filled or status.canceled:
            self._pending_orders.pop(order.order_id, None)

    def _record_trade_fill(self, trade: Trade, side: Side, role: str) -> None:
        """Apply a trade fill to this agent's portfolio and trade log."""
        self._filled_trades.append(trade)

        commodity = trade.commodity
        if commodity not in self.positions:
            self.positions[commodity] = 0.0
        if side == Side.BID:
            self.cash -= trade.price * trade.quantity
            self.positions[commodity] += trade.quantity
        else:
            self.cash += trade.price * trade.quantity
            self.positions[commodity] -= trade.quantity

        self._trade_history.append(
            {
                "tick": trade.timestamp,
                "trade_id": trade.trade_id,
                "price": trade.price,
                "quantity": trade.quantity,
                "side": "buy" if side == Side.BID else "sell",
                "role": role,
            }
        )

        tracked_order_id = (
            trade.maker_order_id if role == "maker" else trade.taker_order_id
        )
        tracked_order = self._pending_orders.get(tracked_order_id)
        if tracked_order is not None and (
            tracked_order.is_filled or tracked_order.canceled
        ):
            self._pending_orders.pop(tracked_order_id, None)

    def cancel_all_orders(self) -> List[OrderStatus]:
        """Cancel all pending orders.

        Returns:
            List of cancellation statuses
        """
        model: MarketModel = self.model
        statuses = []

        for order_id in list(self._pending_orders.keys()):
            pending = self._pending_orders.get(order_id)
            order_commodity = pending.commodity if pending else (self.commodities[0] if self.commodities else "")
            cancel_order = model.exchange.create_order(
                agent_id=self.unique_id,
                side=Side.BID,  # Side doesn't matter for cancel
                order_type=OrderType.CANCEL,
                quantity=0,
                commodity=order_commodity,
            )
            cancel_order.order_id = order_id  # Set the order ID to cancel

            status = model.exchange.submit_order(cancel_order)
            statuses.append(status)

        self._pending_orders.clear()
        return statuses

    def get_state(self) -> Dict:
        """Get agent state for metrics collection.

        Returns:
            Dictionary with agent state
        """
        return {
            "agent_id": self.unique_id,
            "commodities": self.commodities,
            "cash": self.cash,
            "positions": dict(self.positions),
            "net_position": sum(self.positions.values()),
            "initial_cash": self.initial_cash,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "equity": self.equity,
            "return_pct": self.return_pct,
            "pending_orders": len(self._pending_orders),
            "filled_trades": len(self._filled_trades),
            "strategy_type": type(self.strategy).__name__ if self.strategy else None,
            "current_news": self.current_news.to_dict() if self.current_news else None,
            "news_events_seen": len(self._news_history),
        }

    def __repr__(self) -> str:
        net_pos = sum(self.positions.values())
        return (
            f"TradingAgent(id={self.unique_id}, cash={self.cash:.2f}, "
            f"net_position={net_pos:.2f}, pnl={self.total_pnl:.2f})"
        )


# Import Strategy for type hints
from .strategy_interface import Strategy, OrderRequest

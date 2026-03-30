"""Momentum trader strategy.

Simple momentum-based strategy: tracks recent midprice history per
commodity and places buy orders when momentum is positive and sell
orders when momentum is negative. Designed to mirror the structure of
`random_trader.py` so it can be dropped into `builtins` and discovered
automatically.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional

from sim.exchange import OrderType, Side
from sim.news import NewsEvent
from sim.agents.strategy_interface import Observation, OrderRequest, Strategy


class MomentumTraderStrategy(Strategy):
    """Momentum trader based on simple lookback returns.

    Parameters:
        order_quantity: base order size
        lookback: number of ticks to compute momentum over
        threshold: fractional return threshold to trigger market orders
        seed: unused (kept for compatibility)
    """

    def __init__(
        self,
        order_quantity: float = 10.0,
        lookback: int = 5,
        threshold: float = 0.002,
        seed: Optional[int] = None,
    ):
        self.order_quantity = order_quantity
        self.lookback = max(1, int(lookback))
        self.threshold = float(threshold)
        # maintain price history per commodity
        self._price_history: Dict[str, collections.deque] = {}
        self._latest_news: Optional[NewsEvent] = None

    def _update_history(self, obs: Observation) -> None:
        mid = obs.midprice or obs.reference_price
        if mid is None:
            return
        dq = self._price_history.setdefault(obs.commodity, collections.deque(maxlen=self.lookback + 1))
        dq.append(mid)

    def _momentum(self, commodity: str) -> Optional[float]:
        dq = self._price_history.get(commodity)
        if not dq or len(dq) < 2:
            return None
        # simple return over lookback window
        old = dq[0]
        new = dq[-1]
        if old == 0:
            return None
        return (new - old) / old

    def act(self, observations: Dict[str, Observation]) -> List[OrderRequest]:
        orders: List[OrderRequest] = []

        for obs in observations.values():
            # update local history
            self._update_history(obs)

            mid = obs.midprice or obs.reference_price
            if mid is None:
                continue

            mom = self._momentum(obs.commodity)
            if mom is None:
                continue

            qty = max(1.0, self.order_quantity)

            # bias from news
            news_adj = self._latest_news.directional_bias if self._latest_news else 0.0

            # combine signals
            signal = mom + (0.5 * news_adj)

            # Strong momentum -> market order to follow momentum
            if abs(signal) >= self.threshold:
                side = Side.BID if signal > 0 else Side.ASK
                # check cash for buys
                if side == Side.BID and obs.cash < qty * mid:
                    continue
                orders.append(
                    OrderRequest(
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        commodity=obs.commodity,
                        price=None,
                    )
                )
                continue

            # Mild momentum -> place passive limit order on favored side
            side = Side.BID if signal > 0 else Side.ASK
            # price offset small fraction of mid
            offset = mid * 0.0005
            price = mid - offset if side == Side.BID else mid + offset

            # ensure not crossing the book badly
            if side == Side.BID and obs.best_bid is not None:
                price = min(price, obs.best_ask - max(mid * 0.0001, 0.01)) if obs.best_ask else price
            if side == Side.ASK and obs.best_ask is not None:
                price = max(price, obs.best_bid + max(mid * 0.0001, 0.01)) if obs.best_bid else price

            # sanity: ensure price positive
            price = max(0.0001, price)

            orders.append(
                OrderRequest(
                    side=side,
                    order_type=OrderType.LIMIT,
                    quantity=qty,
                    commodity=obs.commodity,
                    price=price,
                )
            )

        return orders

    def on_news(self, news: Optional[NewsEvent]) -> None:
        self._latest_news = news

    def reset(self) -> None:
        self._price_history.clear()
        self._latest_news = None

    def refresh_orders(self) -> bool:
        # keep passive orders refreshed periodically
        return True

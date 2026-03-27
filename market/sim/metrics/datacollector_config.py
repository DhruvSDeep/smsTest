"""DataCollector configuration for Mesa integration.

This module provides utilities for configuring Mesa's DataCollector
to collect market and agent metrics.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from mesa.datacollection import DataCollector

from ..exchange import Side


class MetricsDataCollector:
    """DataCollector configuration for market simulation.

    Provides methods to set up and collect:
    - Market metrics (price, spread, volume)
    - Agent metrics (PnL, position, trades)
    - Custom metrics
    """

    def __init__(self, model: "MarketModel"):
        """Initialize metrics collector.

        Args:
            model: The market model
        """
        self.model = model
        self._custom_reports: List[Callable] = []

        # Create DataCollector
        self.datacollector = DataCollector(
            model_reporters=self._get_model_reporters(),
            agent_reporters=self._get_agent_reporters(),
        )

    def _get_model_reporters(self) -> Dict[str, Callable]:
        """Get model-level reporters."""
        return {
            "tick": lambda m: m.tick,
            "best_bid": lambda m: m.exchange.get_order_book(m.commodities[0]).best_bid,
            "best_ask": lambda m: m.exchange.get_order_book(m.commodities[0]).best_ask,
            "midprice": lambda m: m.exchange.get_order_book(m.commodities[0]).midprice,
            "spread": lambda m: m.exchange.get_order_book(m.commodities[0]).spread,
            "bid_volume": lambda m: m.exchange.get_order_book(m.commodities[0]).get_total_volume(Side.BID),
            "ask_volume": lambda m: m.exchange.get_order_book(m.commodities[0]).get_total_volume(Side.ASK),
            "order_count": lambda m: len(m.exchange.get_order_book(m.commodities[0])),
        }

    def _get_agent_reporters(self) -> Dict[str, Callable]:
        """Get agent-level reporters."""
        return {
            "cash": lambda a: a.cash,
            "net_position": lambda a: sum(a.positions.values()),
            "realized_pnl": lambda a: a._realized_pnl,
            "unrealized_pnl": lambda a: a.unrealized_pnl,
            "total_pnl": lambda a: a.total_pnl,
            "equity": lambda a: a.equity,
            "return_pct": lambda a: a.return_pct,
            "pending_orders": lambda a: len(a._pending_orders),
            "filled_trades": lambda a: len(a._filled_trades),
        }

    def add_model_reporter(self, name: str, reporter: Callable) -> None:
        """Add a custom model reporter.

        Args:
            name: Reporter name
            reporter: Callable that takes model and returns value
        """
        self.datacollector.model_reporters[name] = reporter

    def add_agent_reporter(self, name: str, reporter: Callable) -> None:
        """Add a custom agent reporter.

        Args:
            name: Reporter name
            reporter: Callable that takes agent and returns value
        """
        self.datacollector.agent_reporters[name] = reporter

    def collect(self) -> None:
        """Collect metrics for current step."""
        self.datacollector.collect(self.model)

    def get_model_data(self) -> "Optional[pd.DataFrame]":
        """Get model-level data."""
        dataframe = self.datacollector.get_model_vars_dataframe()
        if dataframe.empty:
            return None
        return dataframe

    def get_agent_data(self) -> "Optional[pd.DataFrame]":
        """Get agent-level data."""
        dataframe = self.datacollector.get_agent_vars_dataframe()
        if dataframe.empty:
            return None
        return dataframe

    def export_data(self, filepath: str) -> None:
        """Export collected data to CSV.

        Args:
            filepath: Output file path
        """
        model_data = self.get_model_data()
        agent_data = self.get_agent_data()

        if model_data is not None:
            model_data.to_csv(f"{filepath}_model.csv")

        if agent_data is not None:
            agent_data.to_csv(f"{filepath}_agent.csv")


# Type hint for pandas
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

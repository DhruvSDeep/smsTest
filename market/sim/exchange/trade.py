"""Trade module - re-exports Trade from order module.

This module exists for backwards compatibility and as a clearer
import path for trade-related functionality.
"""

from .order import Trade, OrderStatus

__all__ = ["Trade", "OrderStatus"]

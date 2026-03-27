"""Strategy loader for dynamic strategy loading.

This module provides utilities for loading strategies at runtime,
including support for user-uploaded strategies and hot-swapping.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Type

from .strategy_interface import Strategy, RandomStrategy


class StrategyLoader:
    """Loads strategy classes from various sources.

    Supports:
    - Built-in strategies
    - User-uploaded strategy files
    - Hot-swapping strategies during simulation
    """

    def __init__(self) -> None:
        """Initialize the strategy loader."""
        self._strategies: dict[str, Type[Strategy]] = {}
        self._load_builtins()

    def _load_builtins(self) -> None:
        """Load built-in strategies."""
        from .builtins import (
            EmptyStrategy,
            RandomStrategy,
            MarketMakerStrategy,
            LiquidityTakerStrategy,
            LiquidityMakerStrategy,
            RandomTraderStrategy,
        )

        self.register("empty", EmptyStrategy)
        self.register("random", RandomStrategy)
        self.register("market_maker", MarketMakerStrategy)
        self.register("liquidity_taker", LiquidityTakerStrategy)
        self.register("liquidity_maker", LiquidityMakerStrategy)
        self.register("random_trader", RandomTraderStrategy)

    def register(self, name: str, strategy_class: Type[Strategy]) -> None:
        """Register a strategy class.

        Args:
            name: Name to register under
            strategy_class: Strategy class to register
        """
        self._strategies[name.lower()] = strategy_class

    def get(self, name: str) -> Type[Strategy]:
        """Get a strategy class by name.

        Args:
            name: Strategy name

        Returns:
            Strategy class

        Raises:
            KeyError: If strategy not found
        """
        name_lower = name.lower()
        if name_lower not in self._strategies:
            raise KeyError(
                f"Strategy '{name}' not found. Available: {list(self._strategies.keys())}"
            )
        return self._strategies[name_lower]

    def create(self, name: str, **kwargs) -> Strategy:
        """Create a strategy instance.

        Args:
            name: Strategy name
            **kwargs: Arguments to pass to strategy constructor

        Returns:
            Strategy instance
        """
        strategy_class = self.get(name)
        return strategy_class(**kwargs)

    def load_from_file(
        self, file_path: str, class_name: str = "Strategy"
    ) -> Type[Strategy]:
        """Load a strategy from a Python file.

        Args:
            file_path: Path to the Python file
            class_name: Name of the strategy class

        Returns:
            Loaded strategy class

        Raises:
            FileNotFoundError: If file doesn't exist
            AttributeError: If class not found in file
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {file_path}")

        # Create module spec
        spec = importlib.util.spec_from_file_location(f"strategy_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load strategy from: {file_path}")

        # Load module
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"strategy_{path.stem}"] = module
        spec.loader.exec_module(module)

        # Get class
        if not hasattr(module, class_name):
            raise AttributeError(f"Class '{class_name}' not found in {file_path}")

        strategy_class = getattr(module, class_name)
        if not issubclass(strategy_class, Strategy):
            raise TypeError(f"Loaded class must be a subclass of Strategy")

        return strategy_class

    def load_from_module(
        self, module_name: str, class_name: str = "Strategy"
    ) -> Type[Strategy]:
        """Load a strategy from an installed module.

        Args:
            module_name: Name of the module
            class_name: Name of the strategy class

        Returns:
            Loaded strategy class
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"Cannot import module '{module_name}': {e}")

        if not hasattr(module, class_name):
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_name}'"
            )

        strategy_class = getattr(module, class_name)
        if not issubclass(strategy_class, Strategy):
            raise TypeError(f"Loaded class must be a subclass of Strategy")

        return strategy_class

    def list_strategies(self) -> list[str]:
        """List all available strategy names."""
        return list(self._strategies.keys())

    def create_from_config(self, config: dict) -> Strategy:
        """Create a strategy from a configuration dict.

        Args:
            config: Configuration dict with 'type' and optional 'params'

        Returns:
            Strategy instance
        """
        strategy_type = config.get("type", "empty")
        params = config.get("params", {})

        return self.create(strategy_type, **params)


# Global loader instance
_default_loader: Optional[StrategyLoader] = None


def get_loader() -> StrategyLoader:
    """Get the default strategy loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = StrategyLoader()
    return _default_loader

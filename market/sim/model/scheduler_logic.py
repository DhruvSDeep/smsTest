"""Market environment - fundamentals, noise, and regime logic.

This module provides optional market environment features like
fundamentals, noise processes, and regime switching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import random


@dataclass
class MarketRegime:
    """Represents a market regime/state.

    Attributes:
        name: Regime name (e.g., "trending", "mean_reverting", "volatile")
        drift: Expected price drift per tick
        volatility: Price volatility multiplier
    """

    name: str
    drift: float = 0.0
    volatility: float = 1.0


class MarketEnvironment:
    """Market environment with fundamentals and noise processes.

    Provides optional features:
    - Price fundamentals (fundamental value)
    - Random walk / noise processes
    - Regime switching (trending, mean-reverting, volatile)

    These are optional - the exchange works without them.
    """

    # Default regimes
    REGIMES = {
        "neutral": MarketRegime("neutral", drift=0.0, volatility=1.0),
        "trending_up": MarketRegime("trending_up", drift=0.01, volatility=1.0),
        "trending_down": MarketRegime("trending_down", drift=-0.01, volatility=1.0),
        "volatile": MarketRegime("volatile", drift=0.0, volatility=2.0),
        "calm": MarketRegime("calm", drift=0.0, volatility=0.5),
    }

    def __init__(
        self,
        initial_price: float = 100.0,
        seed: Optional[int] = None,
        enable_fundamentals: bool = False,
        enable_regimes: bool = False,
        regime_change_prob: float = 0.05,
    ):
        """Initialize market environment.

        Args:
            initial_price: Initial fundamental value
            seed: Random seed
            enable_fundamentals: Whether to track fundamentals
            enable_regimes: Whether to use regime switching
            regime_change_prob: Probability of regime change per tick
        """
        self.initial_price = initial_price
        self.current_price = initial_price

        self.enable_fundamentals = enable_fundamentals
        self.enable_regimes = enable_regimes
        self.regime_change_prob = regime_change_prob

        # Fundamental value
        self.fundamental = initial_price

        # Random number generator
        self._rng = random.Random(seed)

        # Current regime
        self._current_regime = self.REGIMES["neutral"]

        # History
        self._price_history: list[float] = [initial_price]
        self._regime_history: list[str] = ["neutral"]

    @property
    def price(self) -> float:
        """Current price (for reference, not used by exchange)."""
        return self.current_price

    @property
    def regime(self) -> MarketRegime:
        """Current regime."""
        return self._current_regime

    def update(self, tick: float, rng: Optional[random.Random] = None) -> None:
        """Update market environment state.

        Args:
            tick: Current tick
            rng: Random number generator (optional)
        """
        if rng is not None:
            self._rng = rng

        # Update fundamental value (random walk)
        if self.enable_fundamentals:
            noise = self._rng.gauss(0, 0.1)
            self.fundamental += noise

        # Regime switching
        if self.enable_regimes:
            if self._rng.random() < self.regime_change_prob:
                self._switch_regime()

        # Apply regime effects to price
        drift = self._current_regime.drift
        volatility = self._current_regime.volatility
        noise = self._rng.gauss(0, 0.1 * volatility)

        self.current_price += drift + noise

        # Record history
        self._price_history.append(self.current_price)
        self._regime_history.append(self._current_regime.name)

    def _switch_regime(self) -> None:
        """Switch to a new regime."""
        regimes = list(self.REGIMES.values())
        # Current regime should have lower probability
        weights = [
            0.3 if r == self._current_regime else 0.7 / (len(regimes) - 1)
            for r in regimes
        ]

        self._current_regime = self._rng.choices(regimes, weights=weights)[0]

    def get_state(self) -> dict:
        """Get environment state.

        Returns:
            Dictionary with environment state
        """
        return {
            "fundamental": self.fundamental if self.enable_fundamentals else None,
            "current_price": self.current_price,
            "regime": self._current_regime.name if self.enable_regimes else "neutral",
            "regime_drift": self._current_regime.drift,
            "regime_volatility": self._current_regime.volatility,
        }

    def set_regime(self, regime_name: str) -> bool:
        """Manually set the regime.

        Args:
            regime_name: Name of the regime

        Returns:
            True if successful, False if regime not found
        """
        if regime_name in self.REGIMES:
            self._current_regime = self.REGIMES[regime_name]
            return True
        return False

    def get_price_history(self, n: Optional[int] = None) -> list[float]:
        """Get price history.

        Args:
            n: Number of recent prices (None for all)

        Returns:
            List of prices
        """
        if n is None:
            return self._price_history.copy()
        return self._price_history[-n:].copy()

    def reset(self) -> None:
        """Reset environment to initial state."""
        self.current_price = self.initial_price
        self.fundamental = self.initial_price
        self._current_regime = self.REGIMES["neutral"]
        self._price_history = [self.initial_price]
        self._regime_history = ["neutral"]

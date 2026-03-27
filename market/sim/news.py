"""News domain models for market simulations.

Structured news lets the outer simulation loop inject interpretable market
information into agents without coupling strategies directly to the loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into the given range."""
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class NewsSignal:
    """Single news characteristic with a normalized severity.

    Severity is normalized to the range ``[-1.0, 1.0]``. The sign represents
    direction and the magnitude represents intensity.
    """

    severity: float = 0.0
    label: str = ""
    rationale: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", _clamp(float(self.severity), -1.0, 1.0))


@dataclass(frozen=True)
class NewsEvent:
    """Structured news event sent by the main loop to agents.

    Attributes:
        event_id: Unique identifier for the event
        tick: Optional tick associated with the event
        headline: Short headline for logs and reports
        summary: Longer descriptive text
        demand: Positive demand means stronger buying interest
        supply: Positive supply means more available supply / sell pressure
        volatility: Positive volatility means wider uncertainty / volatility
        liquidity: Positive liquidity means deeper / easier trading conditions
        confidence: Confidence in ``[0.0, 1.0]``
        metadata: Optional free-form metadata for strategies or reports
    """

    event_id: str
    tick: Optional[float] = None
    headline: str = ""
    summary: str = ""
    demand: NewsSignal = field(default_factory=NewsSignal)
    supply: NewsSignal = field(default_factory=NewsSignal)
    volatility: NewsSignal = field(default_factory=NewsSignal)
    liquidity: NewsSignal = field(default_factory=NewsSignal)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _clamp(float(self.confidence), 0.0, 1.0))

    @property
    def directional_bias(self) -> float:
        """Net directional pressure implied by demand versus supply."""
        return (self.demand.severity - self.supply.severity) * self.confidence

    @property
    def volatility_bias(self) -> float:
        """Net volatility calibration signal."""
        return self.volatility.severity * self.confidence

    @property
    def liquidity_bias(self) -> float:
        """Net liquidity calibration signal."""
        return self.liquidity.severity * self.confidence

    @property
    def intensity(self) -> float:
        """Overall event intensity across all characteristics."""
        components = [
            abs(self.demand.severity),
            abs(self.supply.severity),
            abs(self.volatility.severity),
            abs(self.liquidity.severity),
        ]
        return (sum(components) / len(components)) * self.confidence

    def price_shift(self, scale: float = 0.01) -> float:
        """Return a multiplicative reservation-price shift."""
        return self.directional_bias * scale

    def spread_multiplier(self) -> float:
        """Return a multiplier suitable for quoting spread calibration."""
        multiplier = 1.0
        multiplier += 0.60 * self.volatility_bias
        multiplier += 0.15 * abs(self.directional_bias)
        multiplier -= 0.25 * self.liquidity_bias
        return max(0.25, multiplier)

    def activity_multiplier(self) -> float:
        """Return a multiplier for order frequency and aggressiveness."""
        multiplier = 1.0
        multiplier += 0.35 * abs(self.directional_bias)
        multiplier += 0.30 * self.liquidity_bias
        multiplier += 0.20 * max(self.volatility_bias, 0.0)
        return max(0.10, multiplier)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event for reporting."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "NewsEvent":
        """Create a news event from a configuration dictionary."""

        def parse_signal(value: Any) -> NewsSignal:
            if value is None:
                return NewsSignal()
            if isinstance(value, (int, float)):
                return NewsSignal(severity=float(value))
            if isinstance(value, dict):
                return NewsSignal(
                    severity=float(value.get("severity", 0.0)),
                    label=str(value.get("label", "")),
                    rationale=str(value.get("rationale", "")),
                )
            raise TypeError(f"Unsupported news signal value: {value!r}")

        event_id = (
            payload.get("event_id") or payload.get("id") or payload.get("headline")
        )
        event_id = str(event_id or f"news-{payload.get('tick', 'unknown')}")

        return cls(
            event_id=event_id,
            tick=payload.get("tick"),
            headline=str(payload.get("headline", "")),
            summary=str(payload.get("summary", "")),
            demand=parse_signal(payload.get("demand")),
            supply=parse_signal(payload.get("supply")),
            volatility=parse_signal(payload.get("volatility")),
            liquidity=parse_signal(payload.get("liquidity")),
            confidence=float(payload.get("confidence", 1.0)),
            metadata=dict(payload.get("metadata", {})),
        )

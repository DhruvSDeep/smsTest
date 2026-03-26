"""Configuration system for simulation parameters.

This module provides configuration classes for market parameters,
agent configs, simulation length, and exchange rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ExchangeConfig:
    """Exchange configuration.

    Attributes:
        tick_interval: Time between ticks
        max_order_size: Maximum order quantity
        min_order_size: Minimum order quantity
        price_precision: Decimal precision for prices
    """

    tick_interval: float = 1.0
    max_order_size: float = 1000.0
    min_order_size: float = 0.01
    price_precision: int = 4


@dataclass
class AgentConfig:
    """Agent configuration.

    Attributes:
        strategy: Strategy name
        strategy_params: Parameters for the strategy
        initial_cash: Starting cash
        position_limit: Maximum position size
    """

    strategy: str = "random"
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    initial_cash: float = 10000.0
    position_limit: Optional[float] = None


@dataclass
class CommodityConfig:
    """Configuration for a single commodity.

    Attributes:
        name: Commodity identifier (e.g. "WHEAT", "CRUDE_OIL")
        initial_price: Starting price for this commodity
        initial_spread: Initial bid-ask spread for this commodity
    """

    name: str = "DEFAULT"
    initial_price: float = 100.0
    initial_spread: float = 1.0


@dataclass
class MarketConfig:
    """Market environment configuration.

    Attributes:
        initial_price: Fallback starting price (used when commodities list is empty)
        initial_spread: Fallback initial bid-ask spread
        enable_fundamentals: Whether to track fundamentals
        enable_regimes: Whether to use regime switching
        commodities: Per-commodity configurations (up to 5)
    """

    initial_price: float = 100.0
    initial_spread: float = 1.0
    enable_fundamentals: bool = False
    enable_regimes: bool = False
    regime_change_prob: float = 0.05
    commodities: List[CommodityConfig] = field(default_factory=list)


@dataclass
class SimulationConfig:
    """Simulation configuration.

    This is the main configuration class that combines all settings.

    Attributes:
        seed: Random seed for reproducibility
        max_steps: Maximum number of simulation steps
        num_agents: Number of trading agents
        exchange: Exchange configuration
        agent: Agent configuration (applied to all agents)
        agents: Individual agent configurations (overrides agent)
        market: Market environment configuration
    """

    seed: Optional[int] = 42
    max_steps: int = 1000
    num_agents: int = 10
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    agents: List[AgentConfig] = field(default_factory=list)
    market: MarketConfig = field(default_factory=MarketConfig)

    @classmethod
    def from_dict(cls, config: Dict) -> "SimulationConfig":
        """Create config from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            SimulationConfig instance
        """
        # Extract nested configs
        exchange_config = config.get("exchange", {})
        agent_config = config.get("agent", {})
        agents_config = config.get("agents", [])
        market_config = config.get("market", {})

        # Create nested configs
        exchange = (
            ExchangeConfig(**exchange_config) if exchange_config else ExchangeConfig()
        )
        agent = AgentConfig(**agent_config) if agent_config else AgentConfig()
        agents = [AgentConfig(**item) for item in agents_config]
        commodities_raw = market_config.pop("commodities", [])
        commodities = [CommodityConfig(**c) for c in commodities_raw]
        market = MarketConfig(**market_config) if market_config else MarketConfig()
        market.commodities = commodities

        # Create main config
        return cls(
            seed=config.get("seed"),
            max_steps=config.get("max_steps", 1000),
            num_agents=config.get("num_agents", 10),
            exchange=exchange,
            agent=agent,
            agents=agents,
            market=market,
        )

    def to_dict(self) -> Dict:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "seed": self.seed,
            "max_steps": self.max_steps,
            "num_agents": self.num_agents,
            "exchange": {
                "tick_interval": self.exchange.tick_interval,
                "max_order_size": self.exchange.max_order_size,
                "min_order_size": self.exchange.min_order_size,
                "price_precision": self.exchange.price_precision,
            },
            "agent": {
                "strategy": self.agent.strategy,
                "strategy_params": self.agent.strategy_params,
                "initial_cash": self.agent.initial_cash,
                "position_limit": self.agent.position_limit,
            },
            "agents": [
                {
                    "strategy": agent.strategy,
                    "strategy_params": agent.strategy_params,
                    "initial_cash": agent.initial_cash,
                    "position_limit": agent.position_limit,
                }
                for agent in self.agents
            ],
            "market": {
                "initial_price": self.market.initial_price,
                "initial_spread": self.market.initial_spread,
                "enable_fundamentals": self.market.enable_fundamentals,
                "enable_regimes": self.market.enable_regimes,
                "regime_change_prob": self.market.regime_change_prob,
                "commodities": [
                    {
                        "name": c.name,
                        "initial_price": c.initial_price,
                        "initial_spread": c.initial_spread,
                    }
                    for c in self.market.commodities
                ],
            },
        }

    @classmethod
    def from_json(cls, json_str: str) -> "SimulationConfig":
        """Create config from JSON string.

        Args:
            json_str: JSON configuration string

        Returns:
            SimulationConfig instance
        """
        import json

        return cls.from_dict(json.loads(json_str))

    def to_json(self) -> str:
        """Convert config to JSON string.

        Returns:
            JSON string
        """
        import json

        return json.dumps(self.to_dict(), indent=2)


# Default configuration
DEFAULT_CONFIG = SimulationConfig()


def load_config(path: str) -> SimulationConfig:
    """Load configuration from file.

    Args:
        path: Path to config file (JSON or YAML)

    Returns:
        SimulationConfig instance
    """
    import json

    with open(path, "r") as f:
        if path.endswith(".json"):
            return SimulationConfig.from_dict(json.load(f))
        elif path.endswith((".yaml", ".yml")):
            import yaml

            return SimulationConfig.from_dict(yaml.safe_load(f))
        else:
            raise ValueError(f"Unsupported config file format: {path}")


def save_config(config: SimulationConfig, path: str) -> None:
    """Save configuration to file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    import json

    with open(path, "w") as f:
        if path.endswith(".json"):
            json.dump(config.to_dict(), f, indent=2)
        elif path.endswith((".yaml", ".yml")):
            import yaml

            yaml.dump(config.to_dict(), f)
        else:
            raise ValueError(f"Unsupported config file format: {path}")

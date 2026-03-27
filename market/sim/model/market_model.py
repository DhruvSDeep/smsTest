"""Market Model - the central simulation orchestrator.

This is the main Mesa Model that orchestrates the entire market simulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from mesa import Model

from ..exchange import MatchingEngine
from ..agents import TradingAgent, Strategy, get_loader
from ..news import NewsEvent
from .scheduler_logic import MarketEnvironment

if TYPE_CHECKING:
    from ..agents.strategy_interface import Observation

# Default 5 commodities with initial prices and spreads
DEFAULT_COMMODITIES = [
    {"name": "WHEAT",       "initial_price": 550.0,  "initial_spread": 2.0},
    {"name": "CRUDE_OIL",   "initial_price": 75.0,   "initial_spread": 0.5},
    {"name": "GOLD",        "initial_price": 1950.0, "initial_spread": 5.0},
    {"name": "COPPER",      "initial_price": 385.0,  "initial_spread": 1.0},
    {"name": "NATURAL_GAS", "initial_price": 250.0,  "initial_spread": 2.0},
]


class MarketModel(Model):
    """Market simulation model - the central orchestrator.

    Supports multiple commodities, each with its own independent order book
    and market environment. Agents can trade across multiple commodities.

    Responsibilities:
    - Simulation clock management
    - Tick orchestration
    - Agent registry (add/remove agents, swap strategies)
    - Per-commodity market environment state
    - Metrics collection via DataCollector
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        num_agents: int = 10,
        initial_cash: float = 10000.0,
        initial_price: float = 100.0,
        tick_interval: float = 1.0,
        agent_strategy: str = "random",
        agent_params: Optional[Dict] = None,
        enable_fundamentals: bool = False,
        enable_regimes: bool = False,
        regime_change_prob: float = 0.05,
        commodity_configs: Optional[List[Dict]] = None,
        **kwargs,
    ):
        """Initialize the market model.

        Args:
            seed: Random seed for reproducibility
            num_agents: Number of trading agents (spread round-robin across commodities)
            initial_cash: Starting cash for each agent
            initial_price: Fallback initial price when commodity_configs not provided
            tick_interval: Time between ticks
            agent_strategy: Default strategy for agents
            agent_params: Parameters for strategy
            enable_fundamentals: Whether to track fundamental values
            enable_regimes: Whether to use regime switching
            regime_change_prob: Probability of regime change per tick
            commodity_configs: List of dicts with keys: name, initial_price, initial_spread.
                               Defaults to 5 built-in commodities.
            **kwargs: Additional arguments for Mesa Model
        """
        super().__init__(seed=seed, **kwargs)

        # Simulation clock
        self._tick_count: float = 0.0
        self.tick_interval = tick_interval

        # Commodity configuration
        if commodity_configs:
            self._commodity_configs: List[Dict] = list(commodity_configs)
        else:
            self._commodity_configs = list(DEFAULT_COMMODITIES)

        self.commodities: List[str] = [c["name"] for c in self._commodity_configs]
        self._commodity_prices: Dict[str, float] = {
            c["name"]: c.get("initial_price", initial_price)
            for c in self._commodity_configs
        }
        self._commodity_spreads: Dict[str, float] = {
            c["name"]: c.get("initial_spread", 1.0)
            for c in self._commodity_configs
        }

        # Exchange (one order book per commodity)
        self.exchange = MatchingEngine(self.commodities)

        # Strategy loader
        self.strategy_loader = get_loader()

        # Per-commodity market environments
        self.environments: Dict[str, MarketEnvironment] = {
            c: MarketEnvironment(
                initial_price=self._commodity_prices[c],
                seed=seed,
                enable_fundamentals=enable_fundamentals,
                enable_regimes=enable_regimes,
                regime_change_prob=regime_change_prob,
            )
            for c in self.commodities
        }

        # Agent registry
        self._agents: Dict[int, TradingAgent] = {}
        self._latest_news: Optional[NewsEvent] = None

        # Create agents (each trades all commodities)
        agent_params = agent_params or {}
        for i in range(num_agents):
            strategy_kwargs = dict(agent_params)
            strategy_kwargs.setdefault("seed", self.random.randrange(2**32))
            strategy = self.strategy_loader.create(agent_strategy, **strategy_kwargs)

            agent = TradingAgent(
                model=self,
                strategy=strategy,
                initial_cash=initial_cash,
                commodities=self.commodities,
            )
            self._agents[agent.unique_id] = agent

        # Seed the random number generator
        if seed is not None:
            self.reset_rng(seed)

        # DataCollector for metrics
        self._setup_data_collection()

        # Initial market state
        self._initialize_market()

    def _setup_data_collection(self) -> None:
        """Set up Mesa DataCollector for metrics."""
        from ..metrics.datacollector_config import MetricsDataCollector

        self.datacollector = MetricsDataCollector(self)

    def _initialize_market(self) -> None:
        """Seed each commodity's order book with initial bid/ask orders."""
        from ..exchange import OrderType, Side

        if not self._agents:
            return

        agents_list = list(self._agents.values())

        for commodity in self.commodities:
            initial_price = self._commodity_prices[commodity]
            initial_spread = self._commodity_spreads[commodity]
            half = initial_spread / 2.0

            commodity_agents = agents_list

            bid_prices = [
                initial_price - half,
                initial_price - half * 2,
                initial_price - half * 3,
            ]
            ask_prices = [
                initial_price + half,
                initial_price + half * 2,
                initial_price + half * 3,
            ]

            for i, price in enumerate(bid_prices):
                agent = commodity_agents[i % len(commodity_agents)]
                order = self.exchange.create_order(
                    agent_id=agent.unique_id,
                    side=Side.BID,
                    order_type=OrderType.LIMIT,
                    quantity=10.0,
                    commodity=commodity,
                    price=price,
                )
                self.exchange.submit_order(order)

            for i, price in enumerate(ask_prices):
                agent = commodity_agents[(i + 3) % len(commodity_agents)]
                order = self.exchange.create_order(
                    agent_id=agent.unique_id,
                    side=Side.ASK,
                    order_type=OrderType.LIMIT,
                    quantity=10.0,
                    commodity=commodity,
                    price=price,
                )
                self.exchange.submit_order(order)

    @property
    def tick(self) -> float:
        """Current simulation tick."""
        return self._tick_count

    @property
    def agents(self) -> List[TradingAgent]:
        """Get all trading agents."""
        return list(self._agents.values())

    def step(self) -> None:
        """Execute one simulation step (tick)."""
        self._tick_count += 1
        self.exchange.next_tick()

        # Update all commodity environments
        for env in self.environments.values():
            env.update(self._tick_count, self.random)

        # Activate agents
        for agent in list(self._agents.values()):
            agent.step()

        # Collect metrics after the step
        if self.datacollector is not None:
            self.datacollector.collect()

    def add_agent(
        self,
        strategy: Optional[Strategy] = None,
        initial_cash: float = 10000.0,
        commodities: Optional[List[str]] = None,
    ) -> TradingAgent:
        """Add a new trading agent.

        Args:
            strategy: Trading strategy (optional, uses default if None)
            initial_cash: Starting cash
            commodities: Commodities to trade (defaults to all model commodities)

        Returns:
            The created agent
        """
        if strategy is None:
            strategy = self.strategy_loader.create(
                "random", seed=self.random.randrange(2**32)
            )

        if commodities is None:
            commodities = self.commodities

        agent = TradingAgent(
            model=self,
            strategy=strategy,
            initial_cash=initial_cash,
            commodities=commodities,
        )
        self._agents[agent.unique_id] = agent
        return agent

    @property
    def current_news(self) -> Optional[NewsEvent]:
        """Most recently broadcast structured news event."""
        return self._latest_news

    def broadcast_news(self, news: Optional["NewsEvent | Dict"]) -> Optional[NewsEvent]:
        """Broadcast a structured news event to all agents."""
        event: Optional[NewsEvent]
        if isinstance(news, dict):
            event = NewsEvent.from_dict(news)
        else:
            event = news

        self._latest_news = event
        for agent in self._agents.values():
            agent.receive_news(event)

        return event

    def remove_agent(self, agent_id: int) -> Optional[TradingAgent]:
        """Remove an agent."""
        agent = self._agents.pop(agent_id, None)
        if agent is not None:
            for commodity in agent.commodities:
                ob = self.exchange.get_order_book(commodity)
                for order in ob.get_orders_for_agent(agent_id):
                    ob.remove_order(order.order_id)
            agent._pending_orders.clear()
            agent.remove()

        return agent

    def swap_strategy(self, agent_id: int, strategy: Strategy) -> bool:
        """Swap an agent's strategy (hot swap)."""
        agent = self._agents.get(agent_id)
        if agent is None:
            return False
        agent.set_strategy(strategy)
        return True

    def get_agent(self, agent_id: int) -> Optional[TradingAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_market_state(self, commodity: Optional[str] = None) -> Dict:
        """Get comprehensive market state for a commodity.

        Args:
            commodity: Commodity name (defaults to first commodity)

        Returns:
            Dictionary with market state for the given commodity
        """
        commodity = commodity or self.commodities[0]
        state = self.exchange.get_market_state(commodity)
        state["tick"] = self._tick_count
        state["environment"] = self.environments[commodity].get_state()
        state["num_agents"] = len(self._agents)
        state["news"] = self._latest_news.to_dict() if self._latest_news else None
        return state

    def get_all_market_states(self) -> Dict[str, Dict]:
        """Get market state for all commodities.

        Returns:
            Dict mapping commodity name -> market state dict
        """
        return {c: self.get_market_state(c) for c in self.commodities}

    def get_agent_metrics(self) -> List[Dict]:
        """Get metrics for all agents."""
        return [agent.get_state() for agent in self._agents.values()]

    def get_leaderboard(self) -> List[Dict]:
        """Get agent leaderboard sorted by PnL."""
        leaderboard = self.get_agent_metrics()
        leaderboard.sort(key=lambda x: x["total_pnl"], reverse=True)
        return leaderboard

    def reset(self, keep_agents: bool = True) -> None:
        """Reset the simulation."""
        self.exchange.reset()
        self._tick_count = 0.0

        for env in self.environments.values():
            env.reset()
        self._latest_news = None

        if keep_agents:
            for agent in self._agents.values():
                agent.cash = agent.initial_cash
                agent.positions = {c: 0.0 for c in agent.commodities}
                agent._realized_pnl = 0.0
                agent._trade_history.clear()
                agent._filled_trades.clear()
                agent._pending_orders.clear()
                agent.receive_news(None)
                if agent.strategy is not None:
                    agent.strategy.reset()
        else:
            self._agents.clear()

    def __repr__(self) -> str:
        first = self.commodities[0] if self.commodities else "?"
        ob = self.exchange.get_order_book(first)
        return (
            f"MarketModel(tick={self._tick_count}, "
            f"commodities={self.commodities}, "
            f"agents={len(self._agents)}, "
            f"{first}: bid={ob.best_bid}, ask={ob.best_ask})"
        )

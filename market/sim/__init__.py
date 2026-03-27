"""SIM - Market Simulation Package

A Mesa-based market simulation framework with:
- Exchange subsystem (order book, matching engine)
- Agent layer (Mesa agents, strategies)
- Model (simulation orchestrator)
- Metrics (PnL, market statistics)
- Visualization (charts, tables, order book view)
- Runtime (config, simulation runner, experiment manager)

Usage:
    from sim import MarketModel, run_simulation

    # Create and run a simulation
    model = MarketModel(seed=42, num_agents=10)
    for _ in range(100):
        model.step()

    # Get results
    leaderboard = model.get_leaderboard()
"""

from .model import MarketModel, MarketEnvironment, MarketRegime
from .exchange import (
    MatchingEngine,
    Order,
    OrderStatus,
    OrderType,
    Side,
    Trade,
    OrderBook,
)
from .agents import (
    TradingAgent,
    Strategy,
    Observation,
    OrderRequest,
    EmptyStrategy,
    RandomStrategy,
    StrategyLoader,
    get_loader,
)
from .metrics import (
    MetricsDataCollector,
    MarketStats,
    PnLMetrics,
    calculate_pnl,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
)
from .runtime import (
    SimulationConfig,
    SimulationRunner,
    run_simulation,
    ExperimentManager,
    DEFAULT_CONFIG,
)
from .visualization import (
    PriceChart,
    OrderBookTable,
    LeaderboardTable,
    OrderBookView,
    create_visualization,
)

__version__ = "0.1.0"

__all__ = [
    # Model
    "MarketModel",
    "MarketEnvironment",
    "MarketRegime",
    # Exchange
    "MatchingEngine",
    "Order",
    "OrderStatus",
    "OrderType",
    "Side",
    "Trade",
    "OrderBook",
    # Agents
    "TradingAgent",
    "Strategy",
    "Observation",
    "OrderRequest",
    "EmptyStrategy",
    "RandomStrategy",
    "StrategyLoader",
    "get_loader",
    # Metrics
    "MetricsDataCollector",
    "MarketStats",
    "PnLMetrics",
    "calculate_pnl",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    # Runtime
    "SimulationConfig",
    "SimulationRunner",
    "run_simulation",
    "ExperimentManager",
    "DEFAULT_CONFIG",
    # Visualization
    "PriceChart",
    "OrderBookTable",
    "LeaderboardTable",
    "OrderBookView",
    "create_visualization",
]

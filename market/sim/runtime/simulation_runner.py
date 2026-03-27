"""Simulation runner for executing market simulations.

This module provides utilities for running simulations, including
single runs, batch runs, and parameter sweeps.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Any

from ..model import MarketModel
from .config import SimulationConfig


class SimulationRunner:
    """Executes market simulations.

    Provides methods for:
    - Single simulation run
    - Batch runs
    - Parameter sweeps
    - Seed control
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the runner.

        Args:
            config: Simulation configuration
        """
        self.config = config or SimulationConfig()

    def create_model(self, **kwargs) -> MarketModel:
        """Create a new model instance.

        Args:
            **kwargs: Override config values

        Returns:
            MarketModel instance
        """
        # Merge config with kwargs
        seed = kwargs.get("seed", self.config.seed)
        num_agents = kwargs.get("num_agents", self.config.num_agents)
        initial_cash = kwargs.get("initial_cash", self.config.agent.initial_cash)
        agent_strategy = kwargs.get("agent_strategy", self.config.agent.strategy)
        agent_params = kwargs.get("agent_params", self.config.agent.strategy_params)
        initial_price = kwargs.get("initial_price", self.config.market.initial_price)
        tick_interval = kwargs.get("tick_interval", self.config.exchange.tick_interval)

        commodity_configs = [
            {"name": c.name, "initial_price": c.initial_price, "initial_spread": c.initial_spread}
            for c in self.config.market.commodities
        ] if self.config.market.commodities else None

        model = MarketModel(
            seed=seed,
            num_agents=0 if self.config.agents else num_agents,
            initial_cash=initial_cash,
            initial_price=initial_price,
            tick_interval=tick_interval,
            agent_strategy=agent_strategy,
            agent_params=agent_params,
            enable_fundamentals=self.config.market.enable_fundamentals,
            enable_regimes=self.config.market.enable_regimes,
            regime_change_prob=self.config.market.regime_change_prob,
            commodity_configs=commodity_configs,
        )

        if self.config.agents:
            for agent_config in self.config.agents:
                strategy_kwargs = dict(agent_config.strategy_params)
                strategy_kwargs.setdefault("seed", model.random.randrange(2**32))
                strategy = model.strategy_loader.create(
                    agent_config.strategy, **strategy_kwargs
                )
                model.add_agent(
                    strategy=strategy,
                    initial_cash=agent_config.initial_cash,
                )
            model._initialize_market()

        return model

    def run(
        self,
        max_steps: Optional[int] = None,
        model: Optional[MarketModel] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict:
        """Run a simulation.

        Args:
            max_steps: Maximum steps (overrides config)
            model: Existing model to run (or create new one)
            callbacks: List of callbacks to call each step

        Returns:
            Final state dictionary
        """
        max_steps = max_steps or self.config.max_steps

        # Create model if not provided
        if model is None:
            model = self.create_model()

        # Run simulation
        for step in range(max_steps):
            model.step()

            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(model, step)

        # Get final state
        return {
            "tick": model.tick,
            "market_state": model.get_market_state(),
            "agents": model.get_agent_metrics(),
            "leaderboard": model.get_leaderboard(),
        }

    def run_batch(
        self,
        num_runs: int,
        max_steps: Optional[int] = None,
    ) -> List[Dict]:
        """Run multiple simulations.

        Args:
            num_runs: Number of runs
            max_steps: Maximum steps per run

        Returns:
            List of final states
        """
        results = []

        for i in range(num_runs):
            # Different seed for each run
            seed = (self.config.seed or 0) + i if self.config.seed else None

            model = self.create_model(seed=seed)
            result = self.run(max_steps=max_steps, model=model)
            results.append(result)

        return results

    def parameter_sweep(
        self,
        param_name: str,
        param_values: List[Any],
        max_steps: Optional[int] = None,
        num_runs: int = 1,
    ) -> Dict:
        """Run parameter sweep.

        Args:
            param_name: Name of parameter to sweep
            param_values: List of values to try
            max_steps: Maximum steps per run
            num_runs: Number of runs per value

        Returns:
            Results dictionary keyed by parameter value
        """
        results = {}

        for value in param_values:
            # Set up config for this run
            runner = SimulationRunner(self.config)
            runner.config = SimulationConfig(
                seed=self.config.seed,
                max_steps=max_steps or self.config.max_steps,
                num_agents=self.config.num_agents,
                agent=type(self.config.agent)(
                    **{**self.config.agent.__dict__, param_name: value}
                ),
                market=self.config.market,
            )

            if num_runs == 1:
                model = runner.create_model()
                result = runner.run(max_steps=max_steps, model=model)
                results[value] = result
            else:
                batch_results = runner.run_batch(num_runs=num_runs, max_steps=max_steps)
                results[value] = batch_results

        return results


def run_simulation(
    config: Optional[SimulationConfig] = None,
    max_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Convenience function to run a single simulation.

    Args:
        config: Simulation configuration
        max_steps: Maximum steps
        verbose: Whether to print progress

    Returns:
        Final state dictionary
    """
    runner = SimulationRunner(config)
    model = runner.create_model()
    active_config = runner.config
    agent_count = (
        len(active_config.agents) if active_config.agents else active_config.num_agents
    )

    if verbose:
        print(
            f"Starting simulation: {agent_count} agents, {max_steps or active_config.max_steps} steps"
        )

    result = runner.run(max_steps=max_steps, model=model)

    if verbose:
        print(f"Simulation complete: tick={result['tick']}")
        print("Top performers:")
        for i, agent in enumerate(result["leaderboard"][:3]):
            print(f"  {i+1}. Agent {agent['agent_id']}: PnL={agent['total_pnl']:.2f}")

    return result

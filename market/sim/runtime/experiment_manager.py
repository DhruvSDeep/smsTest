"""Experiment manager for organizing and running experiments.

This module provides the ExperimentManager class for managing
multiple simulation runs, collecting results, and generating reports.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ExperimentResult:
    """Result of a single experiment run.

    Attributes:
        run_id: Unique identifier for this run
        config: Configuration used
        final_tick: Final tick count
        final_market_state: Final market state
        agent_metrics: Final agent metrics
        leaderboard: Final leaderboard
        duration_ms: Run duration in milliseconds
    """

    run_id: str
    config: Dict
    final_tick: float
    final_market_state: Dict
    agent_metrics: List[Dict]
    leaderboard: List[Dict]
    duration_ms: float


@dataclass
class Experiment:
    """An experiment with multiple runs.

    Attributes:
        name: Experiment name
        description: Experiment description
        config: Base configuration
        num_runs: Number of runs
        results: List of run results
    """

    name: str
    description: str = ""
    config: Dict = field(default_factory=dict)
    num_runs: int = 1
    results: List[ExperimentResult] = field(default_factory=list)


class ExperimentManager:
    """Manages experiments and results.

    Provides:
    - Create and run experiments
    - Store and retrieve results
    - Generate reports
    - Compare strategies
    """

    def __init__(self, output_dir: str = "experiments"):
        """Initialize the experiment manager.

        Args:
            output_dir: Directory for saving experiment data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.experiments: List[Experiment] = []

    def create_experiment(
        self,
        name: str,
        description: str = "",
        config: Optional[Dict] = None,
        num_runs: int = 1,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name
            description: Description
            config: Configuration dictionary
            num_runs: Number of runs

        Returns:
            Created experiment
        """
        experiment = Experiment(
            name=name,
            description=description,
            config=config or {},
            num_runs=num_runs,
        )
        self.experiments.append(experiment)
        return experiment

    def run_experiment(
        self,
        experiment: Experiment,
        max_steps: int = 1000,
        verbose: bool = True,
    ) -> Experiment:
        """Run an experiment.

        Args:
            experiment: The experiment to run
            max_steps: Maximum steps per run
            verbose: Whether to print progress

        Returns:
            The experiment with results
        """
        from .simulation_runner import SimulationRunner
        from .config import SimulationConfig

        # Create config from dict
        config = SimulationConfig.from_dict(experiment.config)
        runner = SimulationRunner(config)

        results: List[ExperimentResult] = []
        start_time = datetime.now()

        for run_idx in range(experiment.num_runs):
            if verbose:
                print(f"  Run {run_idx + 1}/{experiment.num_runs}...")

            # Create model with unique seed
            seed = (config.seed or 0) + run_idx if config.seed else None
            model = runner.create_model(seed=seed)

            # Run
            run_start = datetime.now()
            result = runner.run(max_steps=max_steps, model=model)
            run_end = datetime.now()

            # Create result
            exp_result = ExperimentResult(
                run_id=f"{experiment.name}_run_{run_idx}",
                config=experiment.config,
                final_tick=result["tick"],
                final_market_state=result["market_state"],
                agent_metrics=result["agents"],
                leaderboard=result["leaderboard"],
                duration_ms=(run_end - run_start).total_seconds() * 1000,
            )
            results.append(exp_result)

        experiment.results = results

        if verbose:
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            print(f"  Completed in {total_time:.2f}s")

        return experiment

    def compare_strategies(
        self,
        strategy_names: List[str],
        num_runs: int = 5,
        max_steps: int = 1000,
    ) -> Dict[str, List[Dict]]:
        """Compare multiple strategies.

        Args:
            strategy_names: List of strategy names to compare
            num_runs: Number of runs per strategy
            max_steps: Maximum steps per run

        Returns:
            Results dictionary keyed by strategy name
        """
        results: Dict[str, List[Dict]] = {}

        for strategy in strategy_names:
            if verbose:
                print(f"Testing strategy: {strategy}")

            experiment = self.create_experiment(
                name=f"compare_{strategy}",
                description=f"Comparison run for {strategy}",
                config={"agent": {"strategy": strategy}},
                num_runs=num_runs,
            )

            self.run_experiment(experiment, max_steps=max_steps, verbose=verbose)

            # Extract leaderboard PnLs
            pnl_values = []
            for result in experiment.results:
                if result.leaderboard:
                    pnl_values.append(result.leaderboard[0]["total_pnl"])

            results[strategy] = pnl_values

        return results

    def generate_report(self, experiment: Experiment) -> str:
        """Generate a text report for an experiment.

        Args:
            experiment: The experiment

        Returns:
            Report string
        """
        lines = [
            f"Experiment: {experiment.name}",
            f"Description: {experiment.description}",
            f"Number of runs: {experiment.num_runs}",
            f"Configuration: {json.dumps(experiment.config, indent=2)}",
            "",
            "Results:",
        ]

        # Aggregate metrics
        total_pnls = []
        win_rates = []

        for result in experiment.results:
            if result.leaderboard:
                total_pnls.append(result.leaderboard[0]["total_pnl"])
                win_count = sum(1 for a in result.agent_metrics if a["total_pnl"] > 0)
                win_rates.append(win_count / len(result.agent_metrics) * 100)

        if total_pnls:
            avg_pnl = sum(total_pnls) / len(total_pnls)
            max_pnl = max(total_pnls)
            min_pnl = min(total_pnls)
            avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0

            lines.extend(
                [
                    f"  Average PnL: {avg_pnl:.2f}",
                    f"  Max PnL: {max_pnl:.2f}",
                    f"  Min PnL: {min_pnl:.2f}",
                    f"  Average Win Rate: {avg_win_rate:.1f}%",
                ]
            )

        return "\n".join(lines)

    def save_experiment(
        self, experiment: Experiment, filepath: Optional[str] = None
    ) -> None:
        """Save experiment results to file.

        Args:
            experiment: The experiment to save
            filepath: Output file path (auto-generated if None)
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.output_dir / f"{experiment.name}_{timestamp}.json"

        # Convert to serializable format
        data = {
            "name": experiment.name,
            "description": experiment.description,
            "config": experiment.config,
            "num_runs": experiment.num_runs,
            "results": [
                {
                    "run_id": r.run_id,
                    "config": r.config,
                    "final_tick": r.final_tick,
                    "final_market_state": r.final_market_state,
                    "agent_metrics": r.agent_metrics,
                    "leaderboard": r.leaderboard,
                    "duration_ms": r.duration_ms,
                }
                for r in experiment.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_experiment(self, filepath: str) -> Experiment:
        """Load experiment from file.

        Args:
            filepath: Path to experiment file

        Returns:
            Loaded experiment
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        results = [
            ExperimentResult(
                run_id=r["run_id"],
                config=r["config"],
                final_tick=r["final_tick"],
                final_market_state=r["final_market_state"],
                agent_metrics=r["agent_metrics"],
                leaderboard=r["leaderboard"],
                duration_ms=r["duration_ms"],
            )
            for r in data.get("results", [])
        ]

        return Experiment(
            name=data["name"],
            description=data.get("description", ""),
            config=data.get("config", {}),
            num_runs=data.get("num_runs", 1),
            results=results,
        )


# Module-level verbose flag
verbose = True

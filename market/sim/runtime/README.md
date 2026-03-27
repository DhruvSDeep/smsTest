# Runtime Module

This folder contains configuration, execution, and experiment-management utilities.

## Purpose

- Parse and serialize simulation configuration
- Create and run models programmatically
- Support repeat runs, sweeps, and experiment comparison workflows

## Files

- `config.py`: Dataclass-based configuration model and JSON/YAML helpers.
- `simulation_runner.py`: High-level runner for single runs, batches, and parameter sweeps.
- `experiment_manager.py`: Experiment container and reporting utilities for multiple runs.

## Main Entry Points

- `SimulationConfig`: Main runtime configuration object.
- `SimulationRunner`: Helper for building and executing models.
- `run_simulation(...)`: Convenience wrapper for quick programmatic runs.
- `ExperimentManager`: Organizes named multi-run experiments.

## When To Use This Folder

- Use it when running simulations from Python instead of the `visualise.py` script.
- Use it for reproducible batch experiments or parameter sweeps.
- Use it for loading and saving simulation settings outside the main script.

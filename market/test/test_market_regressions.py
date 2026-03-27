import unittest

from visualise import create_model_from_config, load_config, trade_to_dict
from visualise import run_simulation as run_visual_simulation
from sim.agents.strategy_interface import Observation, RandomStrategy
from sim.exchange import MatchingEngine, OrderType, Side
from sim.metrics import calculate_order_flow
from sim.metrics.datacollector_config import MetricsDataCollector
from sim.runtime.config import SimulationConfig
from sim.runtime.simulation_runner import SimulationRunner, run_simulation


class MatchingEngineTests(unittest.TestCase):
    def test_limit_order_partial_fill_respects_taker_quantity(self):
        engine = MatchingEngine()
        engine.next_tick()

        ask = engine.create_order(
            agent_id=1,
            side=Side.ASK,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            price=101.0,
        )
        engine.submit_order(ask)

        bid = engine.create_order(
            agent_id=2,
            side=Side.BID,
            order_type=OrderType.LIMIT,
            quantity=3.0,
            price=105.0,
        )
        status = engine.submit_order(bid)

        self.assertEqual(status.filled_quantity, 3.0)
        self.assertEqual(status.remaining_quantity, 0.0)
        self.assertEqual(len(status.trades), 1)
        self.assertEqual(status.trades[0].quantity, 3.0)
        self.assertEqual(engine.order_book.get_total_volume(Side.ASK), 7.0)

    def test_trade_to_dict_carries_aggressor_side_for_order_flow(self):
        engine = MatchingEngine()
        engine.next_tick()

        ask = engine.create_order(
            agent_id=1,
            side=Side.ASK,
            order_type=OrderType.LIMIT,
            quantity=5.0,
            price=101.0,
        )
        engine.submit_order(ask)

        bid = engine.create_order(
            agent_id=2,
            side=Side.BID,
            order_type=OrderType.MARKET,
            quantity=5.0,
        )
        status = engine.submit_order(bid)

        trade_dict = trade_to_dict(status.trades[0])
        order_flow = calculate_order_flow([trade_dict])

        self.assertEqual(trade_dict["side"], "buy")
        self.assertEqual(order_flow["buy_volume"], 5.0)
        self.assertEqual(order_flow["sell_volume"], 0)
        self.assertEqual(order_flow["buy_pressure"], 1.0)


class BootstrapAndConfigTests(unittest.TestCase):
    def test_create_model_from_config_does_not_leave_orphan_orders(self):
        config = load_config("config.json")
        model = create_model_from_config(config)

        active_ids = set(model._agents)
        book_agent_ids = {
            order.agent_id for order in model.exchange.order_book._orders_by_id.values()
        }

        self.assertTrue(book_agent_ids)
        self.assertTrue(book_agent_ids.issubset(active_ids))

    def test_simulation_config_preserves_agent_list_and_runner_uses_it(self):
        config = SimulationConfig.from_dict(
            {
                "seed": 7,
                "agents": [
                    {
                        "strategy": "random_trader",
                        "strategy_params": {"probability": 0.0},
                        "initial_cash": 123.0,
                    }
                ],
                "market": {"initial_price": 100.0},
            }
        )

        runner = SimulationRunner(config)
        model = runner.create_model()

        self.assertEqual(len(config.agents), 1)
        self.assertEqual(len(model.agents), 1)
        self.assertEqual(model.agents[0].initial_cash, 123.0)


class HelperApiTests(unittest.TestCase):
    def test_random_strategy_uses_reference_price_when_midprice_missing(self):
        strategy = RandomStrategy(seed=1)
        strategy._random.random = lambda: 0.0
        strategy._random.choice = lambda options: Side.BID
        strategy._random.uniform = lambda a, b: 0.05

        observation = Observation(
            tick=1.0,
            commodity="DEFAULT",
            best_bid=None,
            best_ask=None,
            midprice=None,
            spread=None,
            reference_price=100.0,
            last_trades=[],
            position=0.0,
            cash=10000.0,
            bid_depth=[],
            ask_depth=[],
        )

        orders = strategy.act({"DEFAULT": observation})

        self.assertEqual(len(orders), 1)
        self.assertAlmostEqual(orders[0].price, 100.05)

    def test_metrics_datacollector_returns_dataframes(self):
        model = SimulationRunner().create_model(num_agents=1)
        collector = MetricsDataCollector(model)
        collector.collect()

        model_data = collector.get_model_data()
        agent_data = collector.get_agent_data()

        self.assertIsNotNone(model_data)
        self.assertIsNotNone(agent_data)
        self.assertIn("bid_volume", model_data.columns)
        self.assertIn("ask_volume", model_data.columns)

    def test_run_simulation_accepts_none_config_when_verbose(self):
        result = run_simulation(config=None, max_steps=1, verbose=True)

        self.assertEqual(result["tick"], 1.0)


class BehaviorRegressionTests(unittest.TestCase):
    def test_configured_market_has_broad_participation_and_bounded_book(self):
        config = load_config("config.json")
        model = create_model_from_config(config)
        history = run_visual_simulation(model, 120)

        strategy_groups = {}
        for agent in model.agents:
            state = agent.get_state()
            strategy_groups.setdefault(state["strategy_type"], []).append(state)

        self.assertEqual(
            {
                name: sum(1 for row in rows if row["filled_trades"] == 0)
                for name, rows in strategy_groups.items()
            },
            {
                "MarketMakerStrategy": 0,
                "LiquidityTakerStrategy": 0,
                "LiquidityMakerStrategy": 0,
                "RandomTraderStrategy": 0,
            },
        )

        midprices = [
            round(state["midprice"], 4)
            for state in history
            if state.get("midprice") is not None
        ]
        self.assertGreater(len(set(midprices)), 10)
        self.assertLess(history[-1]["order_count"], 100)


if __name__ == "__main__":
    unittest.main()

"""Safety check module for the market simulation.

#    Run the following command:
#    python -m sim.safety_check

Implements:
1. Taxation - base tax on every trade, extra tax for trades far from market price
2. Collusion detection - flags when two players trade with each other too often
3. VWAP closing price - volume-weighted average price over a time window
4. Deviation spike detection - flags players whose deviation suddenly increases

Usage:
    from sim.safety_check import SafetyCheck

    checker = SafetyCheck(exchange=model.exchange)
    result = checker.process_trade(some_trade)
    report = checker.get_report()
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .exchange.order import Trade, Side
from .exchange.matching_engine import MatchingEngine


class SafetyCheck:
    """Monitors trades for cheating and applies taxation.

    This class does NOT modify any existing files. It reads data from the
    MatchingEngine (to get market prices) and processes Trade objects that
    you pass to it.

    Features:
    1. Taxation: every trade pays a base tax (% of trade value).
       Trades far from market price pay extra tax.
    2. Collusion detection: if two players trade with each other
       way more than with anyone else, flag BOTH players.
    3. VWAP: calculate volume-weighted average closing price over
       a configurable time window (default last 15 ticks).
    4. Deviation spike: track each player's average deviation over time,
       flag if their current deviation is way higher than their average.
    """

    def __init__(
        self,
        exchange: MatchingEngine,
        tax_rate: float = 0.01,
        high_deviation_threshold: float = 0.10,
        high_deviation_tax: float = 0.05,
        collusion_trade_threshold: int = 5,
        collusion_ratio_threshold: float = 0.4,
        collusion_min_total_trades: int = 10,
        vwap_window: float = 15.0,
        spike_multiplier: float = 2.0,
        spike_min_trades: int = 5,
        transfer_value_threshold: float = 5000.0,
    ) -> None:
        """Initialize the safety checker.

        Args:
            exchange: The MatchingEngine instance (to look up market prices)
            tax_rate: Base tax rate as a decimal (0.01 = 1% of trade value)
            high_deviation_threshold: If trade price deviates more than this
                fraction from market price, extra tax applies (0.10 = 10%)
            high_deviation_tax: Extra tax rate for high-deviation trades
                (0.05 = additional 5% on top of base tax)
            collusion_trade_threshold: Minimum number of trades between two
                players before we start checking for collusion
            collusion_ratio_threshold: If more than this fraction of a player's
                trades are with ONE other player, flag it (0.4 = 40%)
            collusion_min_total_trades: Minimum total trades a player must have
                (with everyone, not just one partner) before we check them for
                collusion. Prevents false flags early in the simulation when
                players haven't had time to trade with many people yet.
            vwap_window: Number of ticks to look back for VWAP calculation
                (default 15 = last 15 ticks of the trading day)
            spike_multiplier: Flag if current deviation > average * this number
                (default 2.0 = flag if current deviation is 2x their average)
            spike_min_trades: Minimum trades a player must have before we
                check for spikes (need enough history to have a meaningful average)
            transfer_value_threshold: If the total dollar value of below-threshold
                but still deviated trades between a pair exceeds this amount, flag
                it as a transfer payment. Catches slow money transfers where each
                individual trade looks innocent but the total is massive.
        """
        # Store reference to the exchange so we can look up market prices
        self.exchange = exchange

        # Tax settings
        self.tax_rate = tax_rate
        self.high_deviation_threshold = high_deviation_threshold
        self.high_deviation_tax = high_deviation_tax

        # Collusion settings
        self.collusion_trade_threshold = collusion_trade_threshold
        self.collusion_ratio_threshold = collusion_ratio_threshold
        self.collusion_min_total_trades = collusion_min_total_trades

        # VWAP settings
        self.vwap_window = vwap_window

        # Spike detection settings
        self.spike_multiplier = spike_multiplier
        self.spike_min_trades = spike_min_trades

        # Transfer payment settings
        self.transfer_value_threshold = transfer_value_threshold

        # --- Internal tracking ---

        # Every trade that passes through safety check gets logged here
        self.trade_log: List[Dict] = []

        # Counts how many times each PAIR of players traded with each other
        # Key: tuple of (smaller_id, bigger_id) so (3,7) and (7,3) are the same
        # Value: count of trades between them
        self.pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        # Total trades per player (to calculate ratios)
        self.agent_trade_counts: Dict[int, int] = defaultdict(int)

        # List of all suspicious activity flags
        self.flags: List[Dict] = []

        # Total tax collected from each player
        self.taxes_collected: Dict[int, float] = defaultdict(float)

        # Per-agent deviation history: agent_id -> list of deviation values
        # Used for spike detection (tracking their "normal" behavior)
        self.agent_deviation_history: Dict[int, List[float]] = defaultdict(list)

        # Transfer payment tracking: (agentA, agentB) -> cumulative dollar value
        # of trades that have deviation > 0 but < high_deviation_threshold.
        # When this total exceeds transfer_value_threshold, flag it.
        self.pair_transfer_value: Dict[Tuple[int, int], float] = defaultdict(float)

    def calculate_tax(self, trade: Trade) -> Dict:
        """Calculate tax for a single trade.

        Steps:
        1. Compute trade value = price * quantity
        2. Apply base tax (e.g., 1% of trade value)
        3. Check if trade price is far from market price (deviation)
        4. If deviation is high, add extra tax

        Args:
            trade: The Trade object to tax

        Returns:
            Dict with keys:
                - base_tax: the base tax amount
                - deviation: how far trade price is from market price (as fraction)
                - deviation_tax: extra tax for high deviation (0 if not applicable)
                - total_tax: base_tax + deviation_tax
                - is_high_deviation: True if deviation exceeded threshold
        """
        trade_value = trade.price * trade.quantity
        base_tax = trade_value * self.tax_rate

        # Look up the current market midprice for this commodity
        # midprice = (best_bid + best_ask) / 2
        commodity = trade.commodity
        midprice = None
        if commodity and commodity in self.exchange.order_books:
            midprice = self.exchange.order_books[commodity].midprice

        # Calculate deviation from market price
        deviation = 0.0
        deviation_tax = 0.0
        is_high_deviation = False

        if midprice is not None and midprice > 0:
            # deviation = |trade_price - midprice| / midprice
            # Example: midprice=100, trade at 110 => deviation = 0.10 (10%)
            deviation = abs(trade.price - midprice) / midprice

            if deviation > self.high_deviation_threshold:
                is_high_deviation = True
                # Extra tax on the full trade value
                deviation_tax = trade_value * self.high_deviation_tax

        total_tax = base_tax + deviation_tax

        return {
            "base_tax": round(base_tax, 4),
            "deviation": round(deviation, 4),
            "deviation_tax": round(deviation_tax, 4),
            "total_tax": round(total_tax, 4),
            "is_high_deviation": is_high_deviation,
            "midprice": midprice,
        }

    def check_collusion(self, agent_a: int, agent_b: int, tick: float) -> List[Dict]:
        """Check if two players might be colluding. Flags BOTH players.

        Logic:
        1. Look up how many times these two have traded together
        2. If it's above the threshold (default 5), check the ratio
        3. ratio = (trades between these two) / (total trades by the player)
        4. If ratio > 40% for EITHER player, flag THAT player
        5. Check BOTH players (don't stop after the first)

        Args:
            agent_a: First player's ID
            agent_b: Second player's ID
            tick: Current simulation tick (for the flag timestamp)

        Returns:
            List of flag dicts (one per flagged player, can be 0, 1, or 2)
        """
        # Always store the pair with smaller ID first so (3,7) == (7,3)
        pair = (min(agent_a, agent_b), max(agent_a, agent_b))
        pair_trade_count = self.pair_counts[pair]

        # Don't flag unless they've traded enough times with each other
        if pair_trade_count < self.collusion_trade_threshold:
            return []

        # Don't flag unless BOTH players have enough total trades.
        # Early in the simulation, everyone's ratio looks suspicious because
        # they haven't had time to trade with many different people yet.
        if (self.agent_trade_counts[agent_a] < self.collusion_min_total_trades
                or self.agent_trade_counts[agent_b] < self.collusion_min_total_trades):
            return []

        # Check ratio for EACH player in the pair — don't stop early
        collusion_flags = []
        for agent_id in [agent_a, agent_b]:
            total_trades = self.agent_trade_counts[agent_id]
            if total_trades == 0:
                continue

            ratio = pair_trade_count / total_trades

            if ratio > self.collusion_ratio_threshold:
                other_id = agent_a if agent_id == agent_b else agent_b
                flag = {
                    "type": "collusion_suspected",
                    "tick": tick,
                    "agents": [agent_a, agent_b],
                    "pair_trades": pair_trade_count,
                    "agent_flagged": agent_id,
                    "agent_total_trades": total_trades,
                    "ratio": round(ratio, 4),
                    "message": (
                        f"Agent {agent_id} has traded {pair_trade_count} times "
                        f"with agent {other_id} "
                        f"out of {total_trades} total trades "
                        f"(ratio={ratio:.1%}, threshold={self.collusion_ratio_threshold:.0%})"
                    ),
                }
                self.flags.append(flag)
                collusion_flags.append(flag)

        return collusion_flags

    def check_deviation_spike(
        self, agent_id: int, current_deviation: float, tick: float
    ) -> Optional[Dict]:
        """Check if a player's deviation suddenly spiked above their average.

        We track every player's deviation on every trade. If their current
        deviation is way higher than their historical average (e.g. 2x), flag it.

        This catches players who trade normally for a while, then suddenly
        start making suspicious trades far from market price.

        Args:
            agent_id: The player's ID
            current_deviation: How far this trade's price was from midprice
            tick: Current simulation tick

        Returns:
            A flag dict if spike detected, None otherwise
        """
        history = self.agent_deviation_history[agent_id]

        # Need enough history to compute a meaningful average
        if len(history) < self.spike_min_trades:
            return None

        # Calculate the player's average deviation across all their past trades
        avg_deviation = sum(history) / len(history)

        # If their average is nearly zero, any deviation looks like a huge spike.
        # Only flag if the average itself is meaningful (> 1%)
        if avg_deviation < 0.01:
            # Their average is tiny. Just check if current is above the
            # absolute threshold instead.
            if current_deviation > self.high_deviation_threshold:
                flag = {
                    "type": "deviation_spike",
                    "tick": tick,
                    "agent_id": agent_id,
                    "current_deviation": round(current_deviation, 4),
                    "avg_deviation": round(avg_deviation, 4),
                    "message": (
                        f"Agent {agent_id}: deviation spiked to {current_deviation:.1%} "
                        f"from near-zero average ({avg_deviation:.1%})"
                    ),
                }
                self.flags.append(flag)
                return flag
            return None

        # Check if current deviation is spike_multiplier times their average
        # Example: avg = 3%, current = 8%, multiplier = 2.0
        #   8% > 3% * 2.0 = 6% => YES, this is a spike
        if current_deviation > avg_deviation * self.spike_multiplier:
            flag = {
                "type": "deviation_spike",
                "tick": tick,
                "agent_id": agent_id,
                "current_deviation": round(current_deviation, 4),
                "avg_deviation": round(avg_deviation, 4),
                "spike_ratio": round(current_deviation / avg_deviation, 2),
                "message": (
                    f"Agent {agent_id}: deviation spiked to {current_deviation:.1%} "
                    f"(avg={avg_deviation:.1%}, "
                    f"{current_deviation / avg_deviation:.1f}x their normal)"
                ),
            }
            self.flags.append(flag)
            return flag

        return None

    def check_transfer_payment(
        self, trade: Trade, deviation: float, tick: float
    ) -> Optional[Dict]:
        """Detect slow transfer payments hiding below the deviation threshold.

        Our high-deviation check catches trades >10% off market price. But
        two friends can trade at 4% off (below threshold) with HUGE volume.
        Each trade looks innocent, but the total money moved is massive.

        This method tracks the cumulative dollar value of below-threshold
        deviated trades between each pair. If the total exceeds the
        threshold, flag it.

        Args:
            trade: The trade to check
            deviation: How far the trade price was from midprice (as fraction)
            tick: Current tick

        Returns:
            Flag dict if transfer detected, None otherwise
        """
        # Only care about trades that HAVE some deviation but are BELOW
        # the high threshold (the sneaky range)
        if deviation <= 0.001 or deviation >= self.high_deviation_threshold:
            return None

        pair = (
            min(trade.maker_agent_id, trade.taker_agent_id),
            max(trade.maker_agent_id, trade.taker_agent_id),
        )
        trade_value = trade.price * trade.quantity
        # The "transferred amount" is roughly deviation * trade_value
        # e.g. 4% deviation on a $10,000 trade = $400 transferred
        transferred = deviation * trade_value
        self.pair_transfer_value[pair] += transferred

        if self.pair_transfer_value[pair] > self.transfer_value_threshold:
            flag = {
                "type": "transfer_payment",
                "tick": tick,
                "agents": list(pair),
                "cumulative_transferred": round(self.pair_transfer_value[pair], 2),
                "this_trade_value": round(trade_value, 2),
                "this_trade_deviation": round(deviation, 4),
                "message": (
                    f"Agents {pair[0]} & {pair[1]}: cumulative transfer "
                    f"${self.pair_transfer_value[pair]:.0f} "
                    f"(threshold=${self.transfer_value_threshold:.0f}). "
                    f"Trades are below {self.high_deviation_threshold:.0%} "
                    f"deviation but large volume."
                ),
            }
            self.flags.append(flag)
            return flag

        return None

    def process_trade(self, trade: Trade, model: object = None) -> Dict:
        """Process a trade through all safety checks.

        This is the main method you call for every trade.

        Steps:
        1. Calculate tax
        2. Deduct tax from player cash (if model is provided)
        3. Log the trade
        4. Update pair counts (who traded with who)
        5. Update per-player trade counts
        6. Check for collusion (flags BOTH players)
        7. Check for deviation spikes
        8. Record deviation history
        9. Return results

        Args:
            trade: The Trade object to check
            model: Optional MarketModel instance. If provided, tax is actually
                   deducted from the maker and taker's cash balances.
                   If None, tax is only tracked (not deducted).

        Returns:
            Dict with keys:
                - trade_id: the trade's ID
                - tax: the tax breakdown dict
                - collusion_flags: list of flag dicts (one per flagged player)
                - spike_flag: flag dict if deviation spike detected, else None
        """
        # Step 1: Calculate tax
        tax_info = self.calculate_tax(trade)

        # Step 2: Tax the SELLER only (not the buyer).
        # taker_side tells us what the taker did:
        #   taker_side == ASK means taker is selling => taker is the seller
        #   taker_side == BID means taker is buying  => maker is the seller
        full_tax = tax_info["total_tax"]
        if trade.taker_side == Side.ASK:
            seller_id = trade.taker_agent_id
        else:
            seller_id = trade.maker_agent_id
        self.taxes_collected[seller_id] += full_tax

        # Actually deduct tax from seller's cash if model is provided
        if model is not None:
            seller = model.get_agent(seller_id)
            if seller is not None:
                seller.cash -= full_tax

        # Step 3: Log the trade
        log_entry = {
            "trade_id": trade.trade_id,
            "tick": trade.timestamp,
            "commodity": trade.commodity,
            "maker_id": trade.maker_agent_id,
            "taker_id": trade.taker_agent_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "tax": tax_info,
        }
        self.trade_log.append(log_entry)

        # Step 4: Update pair counts and per-player counts
        pair = (
            min(trade.maker_agent_id, trade.taker_agent_id),
            max(trade.maker_agent_id, trade.taker_agent_id),
        )
        self.pair_counts[pair] += 1
        self.agent_trade_counts[trade.maker_agent_id] += 1
        self.agent_trade_counts[trade.taker_agent_id] += 1

        # Step 5: Check for collusion (now returns list, flags BOTH players)
        collusion_flags = self.check_collusion(
            trade.maker_agent_id, trade.taker_agent_id, trade.timestamp
        )

        # Step 6: If high deviation, create a flag for it
        if tax_info["is_high_deviation"]:
            deviation_flag = {
                "type": "high_deviation_trade",
                "tick": trade.timestamp,
                "commodity": trade.commodity,
                "trade_id": trade.trade_id,
                "maker_id": trade.maker_agent_id,
                "taker_id": trade.taker_agent_id,
                "price": trade.price,
                "deviation": tax_info["deviation"],
                "message": (
                    f"Trade {trade.trade_id} on {trade.commodity}: price={trade.price} "
                    f"deviates {tax_info['deviation']:.1%} from market"
                ),
            }
            self.flags.append(deviation_flag)

        # Step 7: Check for deviation spike per player involved
        deviation = tax_info["deviation"]
        spike_flag = None
        for agent_id in [trade.maker_agent_id, trade.taker_agent_id]:
            spike = self.check_deviation_spike(agent_id, deviation, trade.timestamp)
            if spike is not None:
                spike_flag = spike  # Return the last spike found

        # Step 8: Record deviation in history AFTER spike check
        # (so current trade doesn't pollute the average used to detect itself)
        self.agent_deviation_history[trade.maker_agent_id].append(deviation)
        self.agent_deviation_history[trade.taker_agent_id].append(deviation)

        # Step 9: Check for transfer payments (high volume at low deviation)
        transfer_flag = self.check_transfer_payment(trade, deviation, trade.timestamp)

        return {
            "trade_id": trade.trade_id,
            "tax": tax_info,
            "collusion_flags": collusion_flags,
            "spike_flag": spike_flag,
            "transfer_flag": transfer_flag,
        }

    def calculate_vwap(self, commodity: str, current_tick: float) -> Optional[float]:
        """Calculate Volume-Weighted Average Price for a commodity.

        VWAP = sum(price * quantity) / sum(quantity)
        Only includes trades within the last `vwap_window` ticks.

        This is used as the "closing price" instead of the last trade price,
        so that players can't manipulate the final price by spamming trades
        in the last few minutes.

        Args:
            commodity: Which commodity to calculate VWAP for
            current_tick: The current tick number (to determine the window)

        Returns:
            The VWAP price, or None if no trades in the window
        """
        # Only look at trades in the last N ticks
        window_start = current_tick - self.vwap_window

        total_value = 0.0  # sum of (price * quantity) for each trade
        total_volume = 0.0  # sum of quantity for each trade

        for entry in self.trade_log:
            # Skip trades outside the time window
            if entry["tick"] < window_start:
                continue
            # Skip trades for other commodities
            if entry["commodity"] != commodity:
                continue

            total_value += entry["price"] * entry["quantity"]
            total_volume += entry["quantity"]

        if total_volume == 0:
            return None

        # VWAP = total dollar value / total quantity
        # Example: 3 trades at (100, qty=5), (110, qty=10), (90, qty=5)
        #   total_value = 500 + 1100 + 450 = 2050
        #   total_volume = 5 + 10 + 5 = 20
        #   VWAP = 2050 / 20 = 102.5
        # The trade at 110 pulls the average UP because it had more volume
        return round(total_value / total_volume, 4)

    def get_closing_prices(self, current_tick: float) -> Dict[str, Optional[float]]:
        """Get VWAP closing prices for all commodities.

        Args:
            current_tick: The current tick number

        Returns:
            Dict mapping commodity name -> VWAP price (or None if no trades)
        """
        # Find all commodities that have been traded
        commodities = set(entry["commodity"] for entry in self.trade_log)
        return {c: self.calculate_vwap(c, current_tick) for c in commodities}

    def get_report(self) -> Dict:
        """Get a full safety report.

        Returns:
            Dict with:
                - total_trades: number of trades processed
                - total_tax_collected: sum of all taxes
                - taxes_per_agent: dict of agent_id -> total tax paid
                - flags: list of all suspicious activity flags
                - pair_summary: dict of (agentA, agentB) -> trade count
                    (only pairs with more than 1 trade)
        """
        return {
            "total_trades": len(self.trade_log),
            "total_tax_collected": round(sum(self.taxes_collected.values()), 4),
            "taxes_per_agent": dict(self.taxes_collected),
            "flags": list(self.flags),
            "pair_summary": {
                f"{a}-{b}": count
                for (a, b), count in self.pair_counts.items()
                if count > 1
            },
        }


# ── Test ────────────────────────────────────────────────────────────────
# Run with:  cd smsTest/market && python -m sim.safety_check

def test_safety_check() -> None:
    """Test that creates fake trades, runs them through safety checks,
    and uses assertions to verify correctness."""

    from .exchange.order import Order, OrderType, Side, Trade
    from .exchange.matching_engine import MatchingEngine

    print("=== SafetyCheck Test ===\n")

    # 1. Create a matching engine with one commodity
    engine = MatchingEngine(commodities=["GOLD"])

    # 2. Seed the order book so there's a midprice
    #    We'll place a bid at 495 and ask at 505 => midprice = 500
    bid_order = engine.create_order(
        agent_id=99, side=Side.BID, order_type=OrderType.LIMIT,
        quantity=10.0, commodity="GOLD", price=495.0,
    )
    engine.submit_order(bid_order)

    ask_order = engine.create_order(
        agent_id=98, side=Side.ASK, order_type=OrderType.LIMIT,
        quantity=10.0, commodity="GOLD", price=505.0,
    )
    engine.submit_order(ask_order)

    midprice = engine.order_books["GOLD"].midprice
    print(f"Market midprice for GOLD: {midprice}")
    assert midprice == 500.0, f"Expected midprice 500, got {midprice}"

    # 3. Create the safety checker with 10% deviation threshold
    checker = SafetyCheck(
        exchange=engine,
        tax_rate=0.01,                  # 1% base tax
        high_deviation_threshold=0.10,  # flag trades >10% from midprice
        high_deviation_tax=0.05,        # extra 5% tax on flagged trades
        collusion_trade_threshold=3,    # flag after 3 pair trades
        collusion_ratio_threshold=0.4,  # flag if >40% of trades with one person
        collusion_min_total_trades=10,  # each player needs 10+ total trades first
        spike_min_trades=3,             # need 3 trades before spike detection
        spike_multiplier=2.0,           # flag if 2x their average deviation
    )

    # ── Test 1: Normal trade (low deviation) ──
    print("\n--- Test 1: Normal trade (price=502, midprice=500) ---")
    normal_trade = Trade(
        trade_id=1, maker_order_id=100, taker_order_id=101,
        maker_agent_id=1, taker_agent_id=2,
        price=502.0, quantity=5.0, timestamp=1.0,
        commodity="GOLD",
    )
    result1 = checker.process_trade(normal_trade)
    print(f"  Tax: {result1['tax']['total_tax']}")
    print(f"  Deviation: {result1['tax']['deviation']:.1%}")

    # Assertions: normal trade should NOT be flagged
    assert result1["tax"]["is_high_deviation"] is False, "Normal trade should not be high deviation"
    assert result1["tax"]["deviation_tax"] == 0.0, "Normal trade should have no deviation tax"
    assert result1["tax"]["base_tax"] > 0, "Normal trade should still have base tax"
    assert len(result1["collusion_flags"]) == 0, "First trade should have no collusion flags"
    print("  PASSED: no flags, base tax only")

    # ── Test 2: High deviation trade ──
    print("\n--- Test 2: Suspicious trade (price=600, midprice=500, deviation=20%) ---")
    suspicious_trade = Trade(
        trade_id=2, maker_order_id=102, taker_order_id=103,
        maker_agent_id=3, taker_agent_id=4,
        price=600.0, quantity=5.0, timestamp=2.0,
        commodity="GOLD",
    )
    result2 = checker.process_trade(suspicious_trade)
    print(f"  Tax: {result2['tax']['total_tax']}")
    print(f"  Deviation: {result2['tax']['deviation']:.1%}")

    # Assertions: 20% deviation should be flagged (threshold is 10%)
    assert result2["tax"]["is_high_deviation"] is True, "20% deviation should be flagged"
    assert result2["tax"]["deviation_tax"] > 0, "Should have extra deviation tax"
    assert result2["tax"]["total_tax"] > result1["tax"]["total_tax"], "Suspicious trade tax should be higher"
    print("  PASSED: flagged as high deviation, extra tax applied")

    # ── Test 3: Collusion detection (flags BOTH players) ──
    # With collusion_min_total_trades=10, each player needs 10+ total trades
    # before we even check their collusion ratio. So first give agents 5 & 6
    # some trades with OTHER people (to build up their total trade count),
    # then have them trade with each other.
    print("\n--- Test 3: Collusion (agents 5 & 6, with min total trades check) ---")

    # Phase A: agents 5 and 6 each trade with random other agents (7 trades each)
    # This builds their total_trade_count WITHOUT trading with each other
    print("  Phase A: building trade history with other agents...")
    for i in range(7):
        # Agent 5 trades with agent 50+i (different person each time)
        t = Trade(
            trade_id=100 + i, maker_order_id=1000 + i, taker_order_id=1100 + i,
            maker_agent_id=5, taker_agent_id=50 + i,
            price=500.0, quantity=1.0, timestamp=10.0 + i,
            commodity="GOLD",
        )
        checker.process_trade(t)

        # Agent 6 trades with agent 60+i (different person each time)
        t2 = Trade(
            trade_id=200 + i, maker_order_id=2000 + i, taker_order_id=2100 + i,
            maker_agent_id=6, taker_agent_id=60 + i,
            price=500.0, quantity=1.0, timestamp=10.0 + i,
            commodity="GOLD",
        )
        checker.process_trade(t2)

    print(f"  Agent 5 total trades: {checker.agent_trade_counts[5]}")
    print(f"  Agent 6 total trades: {checker.agent_trade_counts[6]}")

    # Phase B: now agents 5 and 6 trade with EACH OTHER 6 times
    # After phase A, each has 7 trades. After 3 pair trades each will have 10 total.
    # pair_threshold=3, so on the 3rd pair trade (when they also hit 10 total), flag!
    print("  Phase B: agents 5 & 6 now trade with each other...")
    collusion_flags_found = []
    for i in range(6):
        t = Trade(
            trade_id=300 + i, maker_order_id=3000 + i, taker_order_id=3100 + i,
            maker_agent_id=5, taker_agent_id=6,
            price=500.0, quantity=2.0, timestamp=20.0 + i,
            commodity="GOLD",
        )
        result = checker.process_trade(t)
        if result["collusion_flags"]:
            collusion_flags_found.extend(result["collusion_flags"])
            for f in result["collusion_flags"]:
                print(f"  Pair trade {i+1}: FLAGGED agent {f['agent_flagged']}")
        else:
            print(f"  Pair trade {i+1}: no flag yet (total trades: 5={checker.agent_trade_counts[5]}, 6={checker.agent_trade_counts[6]})")

    # Assertions: collusion should eventually be detected
    assert len(collusion_flags_found) > 0, "Should have collusion flags"

    # Both agents 5 AND 6 should be flagged (not just one)
    flagged_agents = set(f["agent_flagged"] for f in collusion_flags_found)
    assert 5 in flagged_agents, "Agent 5 should be flagged for collusion"
    assert 6 in flagged_agents, "Agent 6 should be flagged for collusion"
    print("  PASSED: BOTH agents 5 and 6 flagged (only after enough total trades)")

    # ── Test 4: Tax goes to SELLER only (not buyer) ──
    print("\n--- Test 4: Seller-only tax ---")
    # In the normal_trade (test 1): no taker_side set, defaults to None
    # When taker_side is None (not ASK), seller_id = maker_agent_id = 1
    # So agent 1 (maker/seller) should have ALL the tax, agent 2 (buyer) should have 0
    maker_tax = checker.taxes_collected.get(1, 0)  # agent 1 from normal trade
    buyer_tax = checker.taxes_collected.get(2, 0)   # agent 2 from normal trade
    assert buyer_tax == 0, f"Buyer should NOT be taxed, but has {buyer_tax}"
    assert maker_tax > 0, f"Seller should be taxed, but has {maker_tax}"
    print(f"  Seller (agent 1) tax: {maker_tax}, Buyer (agent 2) tax: {buyer_tax}")
    print("  PASSED: only seller taxed")

    # ── Test 5: VWAP closing price ──
    print("\n--- Test 5: VWAP closing price ---")
    # We've had many trades. Just verify VWAP is not None and is reasonable.
    vwap = checker.calculate_vwap("GOLD", current_tick=30.0)
    assert vwap is not None, "VWAP should not be None (we have trades)"
    # All trades were at 500-600 range, so VWAP should be in that range
    assert 490 < vwap < 610, f"VWAP {vwap} is outside expected range"
    print(f"  VWAP for GOLD: {vwap}")
    print("  PASSED: VWAP calculated correctly")

    # ── Test 6: VWAP ignores trades outside window ──
    print("\n--- Test 6: VWAP window filtering ---")
    # If current_tick=10 and window=5, only trades with tick >= 5 count
    # That's only the collusion trades at tick 3,4,5,6 → tick 5,6 are in window
    vwap_windowed = checker.calculate_vwap("GOLD", current_tick=7.0)
    # window_start = 7.0 - 20.0 = -13.0, so ALL trades are in window
    # Let's use a small window instead
    checker_small_window = SafetyCheck(
        exchange=engine, vwap_window=3.0,
    )
    # Copy trade log from main checker
    checker_small_window.trade_log = list(checker.trade_log)
    vwap_small = checker_small_window.calculate_vwap("GOLD", current_tick=7.0)
    # window_start = 7.0 - 3.0 = 4.0, so only trades at tick >= 4.0 count
    # That's: tick=4.0 (500, qty=2), tick=5.0 (500, qty=2), tick=6.0 (500, qty=2)
    # VWAP = (500*2 + 500*2 + 500*2) / (2+2+2) = 3000/6 = 500
    assert vwap_small == 500.0, f"Expected windowed VWAP 500, got {vwap_small}"
    print(f"  VWAP (window=3, tick=7): {vwap_small}")
    print("  PASSED: window correctly filters old trades")

    # ── Test 7: Deviation spike detection ──
    print("\n--- Test 7: Deviation spike detection ---")
    # Create a new checker for clean state
    checker2 = SafetyCheck(
        exchange=engine,
        high_deviation_threshold=0.10,
        spike_min_trades=3,
        spike_multiplier=2.0,
    )

    # Agent 10 trades with small deviation for 4 trades (building history)
    for i in range(4):
        t = Trade(
            trade_id=50 + i, maker_order_id=500 + i, taker_order_id=600 + i,
            maker_agent_id=10, taker_agent_id=20 + i,  # different partners (no collusion)
            price=510.0, quantity=1.0, timestamp=50.0 + i,  # 2% deviation from 500
            commodity="GOLD",
        )
        result = checker2.process_trade(t)
        assert result["spike_flag"] is None, f"Trade {i}: should not spike yet"

    # Now agent 10 suddenly trades at 650 (30% deviation — way above their avg ~2%)
    spike_trade = Trade(
        trade_id=60, maker_order_id=700, taker_order_id=701,
        maker_agent_id=10, taker_agent_id=25,
        price=650.0, quantity=1.0, timestamp=55.0,
        commodity="GOLD",
    )
    result_spike = checker2.process_trade(spike_trade)
    assert result_spike["spike_flag"] is not None, "Should detect deviation spike"
    print(f"  Agent 10's avg deviation: ~2%, then spiked to 30%")
    print(f"  Spike flag: {result_spike['spike_flag']['message']}")
    print("  PASSED: spike detected")

    # ── Test 8: No collusion for diverse trading ──
    print("\n--- Test 8: No false collusion flags ---")
    checker3 = SafetyCheck(
        exchange=engine,
        collusion_trade_threshold=3,
        collusion_ratio_threshold=0.4,
    )
    # Agent 30 trades with 5 different partners (1 trade each)
    for i in range(5):
        t = Trade(
            trade_id=70 + i, maker_order_id=800 + i, taker_order_id=900 + i,
            maker_agent_id=30, taker_agent_id=40 + i,
            price=500.0, quantity=1.0, timestamp=70.0 + i,
            commodity="GOLD",
        )
        result = checker3.process_trade(t)
        assert len(result["collusion_flags"]) == 0, "Diverse trading should not trigger collusion"
    print("  Agent 30 traded with 5 different partners")
    print("  PASSED: no false collusion flags")

    # ── Test 9: Transfer payment detection ──
    print("\n--- Test 9: Transfer payment (high volume, low deviation) ---")
    checker4 = SafetyCheck(
        exchange=engine,
        high_deviation_threshold=0.10,   # 10% threshold
        transfer_value_threshold=500.0,  # flag when cumulative transfer > $500
    )
    # Agents 80 & 81 trade 20 times at 4% below market (below 10% threshold)
    # Each trade: price=480 (4% below 500), quantity=100
    # Transfer per trade: 4% * 480*100 = 4% * 48000 = $1920
    # After just 1 trade, cumulative = $1920 > $500 threshold => flagged
    transfer_flag_found = None
    for i in range(3):
        t = Trade(
            trade_id=400 + i, maker_order_id=4000 + i, taker_order_id=4100 + i,
            maker_agent_id=80, taker_agent_id=81,
            price=480.0, quantity=100.0, timestamp=80.0 + i,
            commodity="GOLD",
        )
        result = checker4.process_trade(t)
        if result["transfer_flag"] is not None:
            transfer_flag_found = result["transfer_flag"]

    assert transfer_flag_found is not None, "Should flag high-volume low-deviation transfer"
    print(f"  3 trades at 4% deviation with qty=100 each")
    print(f"  {transfer_flag_found['message']}")
    print("  PASSED: transfer payment detected")

    # ── Test 10: No transfer flag for single small trade ──
    print("\n--- Test 10: No false transfer flag for small trades ---")
    checker5 = SafetyCheck(
        exchange=engine,
        high_deviation_threshold=0.10,
        transfer_value_threshold=5000.0,  # high threshold
    )
    # One small trade at 4% deviation: 480 * 1 = $480, transfer = 4% * 480 = $19.2
    small_trade = Trade(
        trade_id=500, maker_order_id=5000, taker_order_id=5100,
        maker_agent_id=90, taker_agent_id=91,
        price=480.0, quantity=1.0, timestamp=90.0,
        commodity="GOLD",
    )
    result = checker5.process_trade(small_trade)
    assert result["transfer_flag"] is None, "Small trade should not trigger transfer flag"
    print("  Small trade at 4% deviation: no flag")
    print("  PASSED: no false transfer flag")

    # ── Final report ──
    report = checker.get_report()
    print("\n=== SAFETY REPORT ===")
    print(f"Total trades processed: {report['total_trades']}")
    print(f"Total tax collected: {report['total_tax_collected']}")
    print(f"Number of flags: {len(report['flags'])}")
    for flag in report["flags"]:
        print(f"  [{flag['type']}] {flag['message']}")

    # Closing prices
    closing = checker.get_closing_prices(current_tick=30.0)
    print(f"\nClosing prices (VWAP): {closing}")

    print("\n=== ALL 10 TESTS PASSED ===")


if __name__ == "__main__":
    test_safety_check()

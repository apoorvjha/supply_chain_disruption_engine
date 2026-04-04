# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Supply Chain Disruption Engine — Environment Implementation.

Topology and all tunable parameters are loaded at runtime from
``supply_chain_disruption_engine/config.yaml`` (or a custom path passed to
``__init__``).  Default network (overridable via config.yaml):

    Suppliers (production sources)   : IDs 0 … num_suppliers-1
    Distribution Centers (warehouses): IDs num_suppliers … num_suppliers+num_dcs-1
    Retailers (demand sinks)         : IDs num_suppliers+num_dcs … total_nodes-1

Agent goal:
    Maximise service level and minimise total supply chain cost across the
    configured episode length while reacting to stochastic disruption events.

Reward signal  (bounded to [0, 1]):

    ref              = sum(initial_inventory) × holding_cost_per_unit × cost_scale_factor
    cost_efficiency  = exp(-step_cost / ref)                               ∈ (0, 1]

    reward = fill_rate_weight      × fill_rate
           + service_level_weight  × service_level
           + cost_efficiency_weight × cost_efficiency                      ∈ [0, 1]

    Weights and cost_scale_factor are set in the ``reward`` section of config.yaml.
"""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..config import SupplyChainConfig, load_config
    from ..models import (
        ActionType,
        DisruptionType,
        INDEX_TO_NODE_ID,
        NodeID,
        SupplyChainDisruptionEngineAction,
        SupplyChainDisruptionEngineObservation,
        build_node_labels,
    )
except ImportError:
    from config import SupplyChainConfig, load_config  # type: ignore[no-redef]
    from models import (  # type: ignore[no-redef]
        ActionType,
        DisruptionType,
        INDEX_TO_NODE_ID,
        NodeID,
        SupplyChainDisruptionEngineAction,
        SupplyChainDisruptionEngineObservation,
        build_node_labels,
    )


class SupplyChainDisruptionEngineEnvironment(Environment):
    """
    A multi-echelon supply chain environment with stochastic disruptions.

    All topology and cost parameters are driven by a YAML configuration file.
    Pass ``config_path`` to ``__init__`` to use a custom file; omit it to use
    the bundled ``supply_chain_disruption_engine/config.yaml``.

    The agent observes the full supply chain state each step and chooses one
    of seven intervention actions (REORDER, EXPEDITE, REROUTE,
    ACTIVATE_BACKUP_SUPPLIER, ADJUST_PRODUCTION, EMERGENCY_PROCUREMENT,
    DO_NOTHING).

    Disruption events are sampled probabilistically and may affect any node.
    The episode terminates after ``episode.max_steps`` steps (configurable).
    """

    # Allow each WebSocket session to get its own isolated environment.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        transform=None,
        rubric=None,
    ):
        """Initialise the environment.

        Args:
            config_path: Path to a YAML config file.  Defaults to the bundled
                ``supply_chain_disruption_engine/config.yaml``.
            transform: Optional observation transform (OpenEnv interface).
            rubric: Optional reward rubric (OpenEnv interface).
        """
        super().__init__(transform=transform, rubric=rubric)
        self._cfg: SupplyChainConfig = load_config(config_path)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random()

        # Warn loudly when a custom topology is used so operators know the
        # web-UI dropdowns (which show default 3-2-4 names) may be incomplete.
        n_s = self._cfg.topology.num_suppliers
        n_d = self._cfg.topology.num_dcs
        n_r = self._cfg.topology.num_retailers
        default_counts = (3, 2, 4)
        if (n_s, n_d, n_r) != default_counts:
            import warnings
            warnings.warn(
                f"Non-default topology detected: {n_s} suppliers, {n_d} DCs, "
                f"{n_r} retailers.  "
                f"The web-UI action dropdowns show names for the default topology "
                f"(3 suppliers, 2 DCs, 4 retailers). "
                f"The environment correctly resolves any 'Supplier-N', 'DC-N', "
                f"'Retailer-N' name at runtime using the loaded config. "
                f"Valid node names are: {self._cfg.all_node_names}",
                stacklevel=2,
            )

        self._init_episode_state()

    # -----------------------------------------------------------------------
    # OpenEnv API — reset()
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupplyChainDisruptionEngineObservation:
        """Reset the supply chain to its configured initial state.

        Args:
            seed: Optional integer seed for reproducible randomness.
            episode_id: Optional custom episode identifier.

        Returns:
            Initial ``SupplyChainDisruptionEngineObservation``.
        """
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._rng = random.Random(seed)
        self._init_episode_state()

        obs = self._build_observation(step_cost=0.0, fill_rate=1.0, done=False, reward=0.0)
        return self._apply_transform(obs)

    # -----------------------------------------------------------------------
    # OpenEnv API — step()
    # -----------------------------------------------------------------------

    def step(
        self,
        action: SupplyChainDisruptionEngineAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SupplyChainDisruptionEngineObservation:
        """Execute one environment step.

        Event order each step:
            1. Apply the agent's action (REORDER, REROUTE, …).
            2. Sample / advance stochastic disruption events.
            3. Deliver in-transit shipments whose countdown has reached zero.
            4. DC → Retailer auto-replenishment (safety-stock pull).
            5. Realise consumer demand at retailers; accumulate backlog.
            6. Compute cost and reward.

        Args:
            action: ``SupplyChainDisruptionEngineAction`` chosen by the agent.
            timeout_s: Ignored (provided for interface compatibility).

        Returns:
            ``SupplyChainDisruptionEngineObservation`` with the new state.
        """
        self._state.step_count += 1

        action_cost = self._apply_action(action)
        self._tick_disruptions()
        self._deliver_orders()
        self._dc_to_retailer_replenishment()
        fill_rate = self._realise_demand()

        costs = self._cfg.costs
        holding_cost = sum(self._inventory) * costs.holding_cost_per_unit
        backlog_cost = sum(self._backlog) * costs.backlog_penalty_per_unit
        step_cost = holding_cost + backlog_cost + action_cost
        self._total_cost += step_cost

        # ── Normalised three-component reward ────────────────────────────
        # 1. Immediate fill rate:  what fraction of this step's demand was met
        # 2. Cumulative service level: overall demand fulfilment since reset
        # 3. Cost efficiency: exponential decay relative to baseline holding cost
        #    → exp(-step_cost / ref)  approaches 1 when cost is low,  0 when high
        rew_cfg = self._cfg.reward
        current_service_level = (
            self._total_fulfilled / self._total_demand
            if self._total_demand > 0.0
            else 1.0
        )
        reference_cost = (
            sum(self._cfg.initial_inventory)
            * costs.holding_cost_per_unit
            * rew_cfg.cost_scale_factor
        )
        cost_efficiency = math.exp(-step_cost / max(reference_cost, 1.0))

        reward = (
            rew_cfg.fill_rate_weight * fill_rate
            + rew_cfg.service_level_weight * current_service_level
            + rew_cfg.cost_efficiency_weight * cost_efficiency
        )

        done = self._state.step_count >= self._cfg.episode.max_steps
        obs = self._build_observation(
            step_cost=step_cost, fill_rate=fill_rate, done=done, reward=reward
        )
        obs.reward = self._apply_rubric(action, obs) if self.rubric else reward
        return self._apply_transform(obs)

    # -----------------------------------------------------------------------
    # OpenEnv API — state property
    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return the current lightweight episode state (episode_id, step_count)."""
        return self._state

    # -----------------------------------------------------------------------
    # Internal helpers — initialisation
    # -----------------------------------------------------------------------

    def _init_episode_state(self) -> None:
        """Initialise / reset all mutable supply chain variables from config."""
        cfg = self._cfg

        # Per-node inventory ordered [suppliers…, DCs…, retailers…]
        self._inventory: List[float] = cfg.initial_inventory

        # Per-retailer backlog (index 0 … num_retailers-1)
        self._backlog: List[float] = [0.0] * cfg.topology.num_retailers

        # Current demand per retailer (disruptions may mutate this temporarily)
        self._base_demand: List[float] = list(cfg.retailers.base_demand)

        # Per-supplier lead times and production capacities (may be disrupted)
        self._lead_times: List[float] = list(cfg.suppliers.base_lead_times)
        self._supplier_capacity: List[float] = list(cfg.suppliers.base_capacity)

        # In-transit shipment orders
        # Each entry: {"target_node": int, "quantity": float, "remaining_steps": int}
        self._in_transit: List[Dict[str, Any]] = []

        # Active disruption events
        self._active_disruptions: List[Dict[str, Any]] = []

        # Backup supplier activation flags
        self._backup_active: List[bool] = [False] * cfg.topology.num_suppliers

        # Episode-level performance accumulators
        self._total_cost: float = 0.0
        self._total_demand: float = 0.0
        self._total_fulfilled: float = 0.0

    # -----------------------------------------------------------------------
    # Internal helpers — action processing
    # -----------------------------------------------------------------------

    def _apply_action(self, action: SupplyChainDisruptionEngineAction) -> float:
        """Execute the agent's action and return the cost incurred (dollars)."""
        atype = action.action_type
        # Resolve node names → integer indices using the *loaded config* so
        # custom topologies (e.g. 4 suppliers, 3 DCs) work without code changes.
        src = self._cfg.resolve_node(action.source_node, fallback_index=0)
        tgt = self._cfg.resolve_node(action.target_node, fallback_index=self._cfg.dc_ids[0])
        qty = action.quantity
        urgency = action.urgency

        if atype == ActionType.DO_NOTHING:
            return 0.0
        if atype == ActionType.REORDER:
            return self._action_reorder(src, tgt, qty, expedite=False)
        if atype == ActionType.EXPEDITE:
            return self._action_reorder(src, tgt, qty, expedite=True)
        if atype == ActionType.REROUTE:
            return self._action_reroute(src, tgt, qty)
        if atype == ActionType.ACTIVATE_BACKUP_SUPPLIER:
            return self._action_activate_backup(src)
        if atype == ActionType.ADJUST_PRODUCTION:
            return self._action_adjust_production(src, urgency)
        if atype == ActionType.EMERGENCY_PROCUREMENT:
            return self._action_emergency_procurement(tgt, qty)
        return 0.0

    def _action_reorder(self, src: int, tgt: int, qty: float, expedite: bool) -> float:
        cfg = self._cfg
        supplier_ids = cfg.supplier_ids
        dc_ids = cfg.dc_ids

        if src not in supplier_ids or tgt not in dc_ids:
            return 0.0

        sup_idx = supplier_ids.index(src)
        shippable = min(qty, self._supplier_capacity[sup_idx], self._inventory[src])
        if shippable <= 0.0:
            return 0.0

        base_lead = self._lead_times[sup_idx]
        lead_time = max(1, int(base_lead * 0.5 if expedite else base_lead))

        self._in_transit.append(
            {"target_node": tgt, "quantity": shippable, "remaining_steps": lead_time}
        )
        self._inventory[src] -= shippable

        costs = cfg.costs
        unit_cost = costs.order_cost_per_unit * (costs.expedite_premium if expedite else 1.0)
        return shippable * unit_cost

    def _action_reroute(self, src: int, tgt: int, qty: float) -> float:
        num_nodes = self._cfg.num_nodes
        if src == tgt or not (0 <= src < num_nodes) or not (0 <= tgt < num_nodes):
            return 0.0

        transferable = min(qty, self._inventory[src])
        if transferable <= 0.0:
            return 0.0

        self._inventory[src] -= transferable
        self._inventory[tgt] += transferable

        costs = self._cfg.costs
        return costs.reroute_cost_flat + transferable * costs.reroute_cost_per_unit

    def _action_activate_backup(self, src: int) -> float:
        cfg = self._cfg
        supplier_ids = cfg.supplier_ids
        if src not in supplier_ids:
            return 0.0

        sup_idx = supplier_ids.index(src)
        if self._backup_active[sup_idx]:
            return 0.0

        self._backup_active[sup_idx] = True
        fraction = cfg.suppliers.backup_capacity_fraction
        self._supplier_capacity[sup_idx] = max(
            self._supplier_capacity[sup_idx],
            cfg.suppliers.base_capacity[sup_idx] * fraction,
        )
        return cfg.costs.activate_backup_cost

    def _action_adjust_production(self, src: int, urgency: float) -> float:
        cfg = self._cfg
        supplier_ids = cfg.supplier_ids
        if src not in supplier_ids:
            return 0.0

        sup_idx = supplier_ids.index(src)
        max_cap = cfg.suppliers.base_capacity[sup_idx] * cfg.costs.production_ramp_max_fraction
        boost = 1.0 + urgency * (cfg.costs.production_ramp_max_fraction - 1.0)
        new_cap = min(max_cap, self._supplier_capacity[sup_idx] * boost)
        delta = max(0.0, new_cap - self._supplier_capacity[sup_idx])
        self._supplier_capacity[sup_idx] = new_cap
        return delta * cfg.costs.production_ramp_cost_per_unit

    def _action_emergency_procurement(self, tgt: int, qty: float) -> float:
        cfg = self._cfg
        if not (0 <= tgt < cfg.num_nodes) or qty <= 0.0:
            return 0.0

        self._in_transit.append({"target_node": tgt, "quantity": qty, "remaining_steps": 1})
        return qty * cfg.costs.order_cost_per_unit * cfg.costs.emergency_premium

    # -----------------------------------------------------------------------
    # Internal helpers — disruption dynamics
    # -----------------------------------------------------------------------

    def _tick_disruptions(self) -> None:
        """Advance active disruptions; remove expired ones; possibly spawn new."""
        still_active: List[Dict[str, Any]] = []
        for d in self._active_disruptions:
            d["remaining_steps"] -= 1
            if d["remaining_steps"] > 0:
                still_active.append(d)
            else:
                self._recover_from_disruption(d)
        self._active_disruptions = still_active

        dis_cfg = self._cfg.disruptions
        if (
            len(self._active_disruptions) < dis_cfg.max_concurrent
            and self._rng.random() < dis_cfg.probability_per_step
        ):
            new_d = self._sample_disruption()
            self._apply_disruption(new_d)
            self._active_disruptions.append(new_d)

    def _sample_disruption(self) -> Dict[str, Any]:
        cfg = self._cfg
        dis_cfg = cfg.disruptions
        dtype = self._rng.choice(list(DisruptionType))
        affected_node = self._rng.randint(0, cfg.num_nodes - 1)
        severity = round(self._rng.uniform(dis_cfg.min_severity, dis_cfg.max_severity), 3)
        duration = self._rng.randint(dis_cfg.min_duration, dis_cfg.max_duration)
        return {
            "type": dtype.value,
            "affected_node": affected_node,                          # int index internally
            "affected_node_name": INDEX_TO_NODE_ID.get(affected_node, str(affected_node)),
            "severity": severity,
            "remaining_steps": duration,
            "total_duration": duration,
        }

    def _apply_disruption(self, d: Dict[str, Any]) -> None:
        node = d["affected_node"]
        sev = d["severity"]
        dtype = d["type"]
        cfg = self._cfg
        supplier_ids = cfg.supplier_ids
        retailer_ids = cfg.retailer_ids

        if dtype == DisruptionType.NATURAL_DISASTER.value:
            self._inventory[node] *= max(0.0, 1.0 - sev)
            if node in supplier_ids:
                sup_idx = supplier_ids.index(node)
                self._supplier_capacity[sup_idx] *= max(0.0, 1.0 - sev)

        elif dtype in (
            DisruptionType.TRANSPORTATION_FAILURE.value,
            DisruptionType.PORT_CONGESTION.value,
        ):
            if node in supplier_ids:
                sup_idx = supplier_ids.index(node)
                self._lead_times[sup_idx] = min(
                    cfg.suppliers.base_lead_times[sup_idx] * (1.0 + sev * 2.5),
                    cfg.episode.max_steps,
                )

        elif dtype == DisruptionType.LABOR_STRIKE.value:
            if node in supplier_ids:
                sup_idx = supplier_ids.index(node)
                self._supplier_capacity[sup_idx] *= max(0.0, 1.0 - sev * 0.85)

        elif dtype == DisruptionType.DEMAND_SPIKE.value:
            if node in retailer_ids:
                ret_idx = retailer_ids.index(node)
                self._base_demand[ret_idx] *= 1.0 + sev

        elif dtype == DisruptionType.SUPPLIER_BANKRUPTCY.value:
            if node in supplier_ids:
                sup_idx = supplier_ids.index(node)
                self._supplier_capacity[sup_idx] = 0.0

    def _recover_from_disruption(self, d: Dict[str, Any]) -> None:
        node = d["affected_node"]
        dtype = d["type"]
        cfg = self._cfg
        supplier_ids = cfg.supplier_ids
        retailer_ids = cfg.retailer_ids

        if node in supplier_ids:
            sup_idx = supplier_ids.index(node)
            self._lead_times[sup_idx] = cfg.suppliers.base_lead_times[sup_idx]
            fraction = (
                cfg.suppliers.backup_capacity_fraction
                if self._backup_active[sup_idx]
                else 1.0
            )
            self._supplier_capacity[sup_idx] = cfg.suppliers.base_capacity[sup_idx] * fraction
            self._backup_active[sup_idx] = False

        if dtype == DisruptionType.DEMAND_SPIKE.value and node in retailer_ids:
            ret_idx = retailer_ids.index(node)
            self._base_demand[ret_idx] /= 1.0 + d["severity"]

    # -----------------------------------------------------------------------
    # Internal helpers — material flow
    # -----------------------------------------------------------------------

    def _deliver_orders(self) -> None:
        """Advance shipment countdowns; credit arrived orders to target nodes."""
        remaining: List[Dict[str, Any]] = []
        for order in self._in_transit:
            order["remaining_steps"] -= 1
            if order["remaining_steps"] <= 0:
                self._inventory[order["target_node"]] += order["quantity"]
            else:
                remaining.append(order)
        self._in_transit = remaining

    def _dc_to_retailer_replenishment(self) -> None:
        """Push inventory from each DC to its assigned retailers up to safety stock."""
        cfg = self._cfg
        safety_factor = cfg.retailers.safety_stock_days
        retailer_dc_map = cfg.retailer_dc_map
        retailer_ids = cfg.retailer_ids

        for ret_idx, dc_id in enumerate(retailer_dc_map):
            node_id = retailer_ids[ret_idx]
            safety_stock = self._base_demand[ret_idx] * safety_factor
            gap = max(0.0, safety_stock - self._inventory[node_id])
            if gap <= 0.0:
                continue
            ship = min(gap, self._inventory[dc_id])
            if ship > 0.0:
                self._inventory[dc_id] -= ship
                self._inventory[node_id] += ship

    def _realise_demand(self) -> float:
        """Simulate consumer demand at each retailer; return step fill rate."""
        cfg = self._cfg
        retailer_ids = cfg.retailer_ids
        noise_std = cfg.retailers.demand_noise_std

        step_demand = 0.0
        step_fulfilled = 0.0

        for ret_idx, node_id in enumerate(retailer_ids):
            noise = self._rng.gauss(0.0, noise_std)
            demand = max(0.0, self._base_demand[ret_idx] + noise)
            step_demand += demand

            obligation = demand + self._backlog[ret_idx]
            fulfilled = min(obligation, max(0.0, self._inventory[node_id]))

            self._inventory[node_id] = max(0.0, self._inventory[node_id] - fulfilled)
            self._backlog[ret_idx] = max(0.0, obligation - fulfilled)
            step_fulfilled += min(fulfilled, demand)

        self._total_demand += step_demand
        self._total_fulfilled += step_fulfilled
        return step_fulfilled / step_demand if step_demand > 0.0 else 1.0

    # -----------------------------------------------------------------------
    # Internal helpers — observation construction
    # -----------------------------------------------------------------------

    def _build_observation(
        self,
        step_cost: float,
        fill_rate: float,
        done: bool,
        reward: float,
    ) -> SupplyChainDisruptionEngineObservation:
        cfg = self._cfg
        service_level = (
            self._total_fulfilled / self._total_demand
            if self._total_demand > 0.0
            else 1.0
        )

        noise_std = cfg.retailers.demand_noise_std * 0.5
        demand_forecast = [
            round(max(0.0, d + self._rng.gauss(0.0, noise_std)), 2)
            for d in self._base_demand
        ]

        i2n = self._cfg.index_to_node_name
        disruptions_snapshot = [
            {
                "type": d["type"],
                "affected_node": i2n.get(d["affected_node"], str(d["affected_node"])),
                "severity": d["severity"],
                "remaining_steps": d["remaining_steps"],
                "total_duration": d["total_duration"],
            }
            for d in self._active_disruptions
        ]

        n_s = cfg.topology.num_suppliers
        n_d = cfg.topology.num_dcs
        n_r = cfg.topology.num_retailers

        return SupplyChainDisruptionEngineObservation(
            inventory_levels=[round(v, 2) for v in self._inventory],
            backlog=[round(v, 2) for v in self._backlog],
            demand_forecast=demand_forecast,
            lead_times=[round(v, 2) for v in self._lead_times],
            supplier_capacity=[round(v, 2) for v in self._supplier_capacity],
            active_disruptions=disruptions_snapshot,
            service_level=round(max(0.0, min(1.0, service_level)), 4),
            fill_rate=round(max(0.0, min(1.0, fill_rate)), 4),
            total_cost=round(self._total_cost, 2),
            step_cost=round(step_cost, 2),
            in_transit_orders=len(self._in_transit),
            done=done,
            reward=float(reward),
            metadata={
                "step": self._state.step_count,
                "episode_id": self._state.episode_id,
                "backup_active": list(self._backup_active),
                "topology": {
                    "num_suppliers": n_s,
                    "num_dcs": n_d,
                    "num_retailers": n_r,
                },
            },
        )


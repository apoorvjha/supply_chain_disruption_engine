# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Supply Chain Disruption Engine Environment.

Node-naming convention (scales with topology):
  Suppliers            → "Supplier-1", "Supplier-2", …, "Supplier-N"
  Distribution Centres → "DC-1", "DC-2", …, "DC-M"
  Retailers            → "Retailer-1", "Retailer-2", …, "Retailer-K"

In the action model, ``source_node`` and ``target_node`` accept any valid
node name for the *currently loaded* topology.  The ``NodeID`` enum below
documents the names for the **default** topology (3 suppliers, 2 DCs,
4 retailers) and is used to generate the web-UI dropdown schema; it is not
used for runtime validation so that custom topologies work without code changes.

Flow:  Suppliers --[agent orders]--> DCs --[auto-replenish]--> Retailers --> Demand
"""

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


# ---------------------------------------------------------------------------
# Node-label builder  (single source of truth for the naming convention)
# ---------------------------------------------------------------------------


def build_node_labels(
    n_suppliers: int,
    n_dcs: int,
    n_retailers: int,
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Generate node names and bidirectional index mappings for any topology.

    Node naming follows the convention:
        Supplier-1 … Supplier-N  (indices 0 … n_suppliers-1)
        DC-1 … DC-M              (indices n_suppliers … n_suppliers+n_dcs-1)
        Retailer-1 … Retailer-K  (indices n_suppliers+n_dcs … total-1)

    Args:
        n_suppliers: Number of supplier nodes.
        n_dcs:       Number of distribution centre nodes.
        n_retailers: Number of retailer nodes.

    Returns:
        A 3-tuple of:
          - ``names``         : Ordered list of all node names.
          - ``name_to_index`` : Dict mapping name → integer index.
          - ``index_to_name`` : Dict mapping integer index → name.

    Example::

        names, n2i, i2n = build_node_labels(3, 2, 4)
        n2i["DC-1"]        # → 3
        i2n[5]             # → "Retailer-1"

        names, n2i, i2n = build_node_labels(4, 3, 5)   # custom topology
        n2i["Supplier-4"]  # → 3
        n2i["DC-3"]        # → 6
        n2i["Retailer-5"]  # → 11
    """
    names: List[str] = (
        [f"Supplier-{i + 1}" for i in range(n_suppliers)]
        + [f"DC-{i + 1}" for i in range(n_dcs)]
        + [f"Retailer-{i + 1}" for i in range(n_retailers)]
    )
    name_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(names)}
    index_to_name: Dict[int, str] = {idx: name for idx, name in enumerate(names)}
    return names, name_to_index, index_to_name


# ---------------------------------------------------------------------------
# Config-driven node-label constants (loaded once at import time)
# ---------------------------------------------------------------------------

# Read the topology counts directly from config.yaml so that NodeID and its
# schema always reflect the *actual* deployment topology — no edits required
# when num_suppliers / num_dcs / num_retailers are changed.
_n_s, _n_d, _n_r = 3, 2, 4  # safe fallback (default topology)
try:
    import yaml as _yaml
    from pathlib import Path as _Path

    _cfg_path = _Path(__file__).parent / "config.yaml"
    if _cfg_path.exists():
        with _cfg_path.open() as _fh:
            _topo = _yaml.safe_load(_fh).get("topology", {})
        _n_s = int(_topo.get("num_suppliers", _n_s))
        _n_d = int(_topo.get("num_dcs", _n_d))
        _n_r = int(_topo.get("num_retailers", _n_r))
except Exception:
    pass  # keep safe fallback values

_ALL_NODE_NAMES, NODE_ID_TO_INDEX, INDEX_TO_NODE_ID = build_node_labels(_n_s, _n_d, _n_r)

# Regex accepts any valid node name regardless of topology:
#   "Supplier-N", "DC-N", "Retailer-N"  where N ≥ 1
_NODE_NAME_RE = re.compile(r"^(Supplier|DC|Retailer)-[1-9]\d*$")

# Dynamic str Enum — always matches the topology from config.yaml.
# Member keys use UPPER_WITH_UNDERSCORE (e.g. SUPPLIER_1, DC_3, RETAILER_4);
# members are accessible as attributes: NodeID.DC_3, NodeID.RETAILER_1, …
#
# Action routing rules
# --------------------
# REORDER / EXPEDITE          : source = Supplier-*, target = DC-*
# REROUTE                     : source = any node,   target = any node (must differ)
# ACTIVATE_BACKUP_SUPPLIER    : source = Supplier-*  (target ignored)
# ADJUST_PRODUCTION           : source = Supplier-*  (target ignored)
# EMERGENCY_PROCUREMENT       : target = any node    (source ignored)
# DO_NOTHING                  : both fields ignored
NodeID: type = Enum(  # type: ignore[assignment]
    "NodeID",
    {name.upper().replace("-", "_"): name for name in _ALL_NODE_NAMES},
    type=str,
)

# Convenience sets — derived from the dynamic enum
SUPPLIER_NODES: frozenset = frozenset(n for n in NodeID if str(n.value).startswith("Supplier-"))
DC_NODES: frozenset = frozenset(n for n in NodeID if str(n.value).startswith("DC-"))
RETAILER_NODES: frozenset = frozenset(n for n in NodeID if str(n.value).startswith("Retailer-"))


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """Available agent action types."""

    DO_NOTHING = "do_nothing"
    """Take no action this step."""

    REORDER = "reorder"
    """Place a replenishment order from a supplier (source_node) to a DC
    (target_node).  Units enter transit and arrive after the supplier's
    current lead time."""

    EXPEDITE = "expedite"
    """Rush an order from a supplier to a DC at a cost premium; lead time is
    halved compared to a standard REORDER."""

    REROUTE = "reroute"
    """Transfer available inventory directly from source_node to target_node
    within the same tier or across tiers (e.g., DC-to-DC or DC-to-retailer)
    at a flat logistics cost."""

    ACTIVATE_BACKUP_SUPPLIER = "activate_backup_supplier"
    """Activate a contingency supplier for source_node. Restores ~70 % of
    baseline capacity when the primary supplier is disrupted.  One-time setup
    cost applies."""

    ADJUST_PRODUCTION = "adjust_production"
    """Temporarily raise the production capacity of source_node (a supplier)
    proportional to the urgency signal. Incurs a variable ramp-up cost."""

    EMERGENCY_PROCUREMENT = "emergency_procurement"
    """Buy inventory on the spot market, delivered to target_node in one
    step.  Highest per-unit cost — reserved for critical stockout situations."""


class DisruptionType(str, Enum):
    """Categories of supply chain disruption events."""

    NATURAL_DISASTER = "natural_disaster"
    """Destroys a fraction of on-hand inventory and reduces node capacity."""

    TRANSPORTATION_FAILURE = "transportation_failure"
    """Increases supplier lead times for the duration of the event."""

    LABOR_STRIKE = "labor_strike"
    """Cuts supplier production capacity until the strike is resolved."""

    DEMAND_SPIKE = "demand_spike"
    """Temporary surge in consumer demand at a retailer node."""

    SUPPLIER_BANKRUPTCY = "supplier_bankruptcy"
    """Supplier capacity drops to zero; recovery requires backup activation."""

    PORT_CONGESTION = "port_congestion"
    """Severe delay multiplier on inbound shipment lead times."""


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class SupplyChainDisruptionEngineAction(Action):
    """Action submitted by the agent each step.

    ``source_node`` and ``target_node`` accept any node name that matches the
    active topology: ``"Supplier-1"``, ``"DC-2"``, ``"Retailer-3"``, etc.
    The web UI renders dropdowns populated from the default-topology NodeID enum;
    custom topologies work at runtime without schema changes.
    """

    @classmethod
    def model_json_schema(cls, **kwargs):
        """Inline enum values so the web UI renders dropdowns for all enum fields.

        ActionType, source_node, and target_node are all inlined from their
        ``$defs`` entry so that the web UI shows a select/dropdown control.
        The NodeID enum is built from the *loaded* ``config.yaml`` topology,
        so the dropdown always lists exactly the nodes that exist.
        """
        schema = super().model_json_schema(**kwargs)
        defs = schema.get("$defs", {})
        props = schema.get("properties", {})

        # Inline each enum field: replace $ref with the inlined enum definition
        for field_name, def_name in [
            ("action_type", "ActionType"),
            ("source_node", "NodeID"),
            ("target_node", "NodeID"),
        ]:
            enum_def = defs.get(def_name, {})
            prop = props.get(field_name, {})
            if enum_def and "enum" in enum_def:
                props[field_name] = {
                    "enum": enum_def["enum"],
                    "type": enum_def.get("type", "string"),
                    "title": prop.get("title", def_name),
                    "default": prop.get("default"),
                    "description": prop.get("description", enum_def.get("description", "")),
                }

        return schema

    action_type: ActionType = Field(
        default=ActionType.DO_NOTHING,
        description="Type of supply chain intervention to execute.",
    )
    source_node: NodeID = Field(  # type: ignore[valid-type]
        default="Supplier-1",
        description=(
            "Origin node — e.g. 'Supplier-2', 'DC-1'. "
            "REORDER / EXPEDITE / ACTIVATE_BACKUP_SUPPLIER / ADJUST_PRODUCTION → "
            "must be a Supplier node. "
            "REROUTE → any node that holds excess inventory. "
            "EMERGENCY_PROCUREMENT / DO_NOTHING → field is ignored."
        ),
    )
    target_node: NodeID = Field(  # type: ignore[valid-type]
        default="DC-1",
        description=(
            "Destination node — e.g. 'DC-1', 'Retailer-3'. "
            "REORDER / EXPEDITE → must be a DC node. "
            "REROUTE / EMERGENCY_PROCUREMENT → any receiving node. "
            "ACTIVATE_BACKUP_SUPPLIER / ADJUST_PRODUCTION / DO_NOTHING → field is ignored."
        ),
    )
    quantity: float = Field(
        default=0.0,
        ge=0.0,
        le=2000.0,
        description="Units to order, transfer, or procure.",
    )
    urgency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Urgency level [0, 1] used by ADJUST_PRODUCTION to determine the "
            "capacity ramp-up multiplier."
        ),
    )

    @field_validator("source_node", "target_node", mode="before")
    @classmethod
    def _validate_node_name(cls, v: Any) -> str:
        """Coerce NodeID enum members and bare strings to the canonical string value."""
        if isinstance(v, Enum):
            return v.value
        s = str(v)
        if not _NODE_NAME_RE.match(s):
            raise ValueError(
                f"Invalid node name {s!r}. "
                "Expected 'Supplier-N', 'DC-N', or 'Retailer-N' (N ≥ 1)."
            )
        return s


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class SupplyChainDisruptionEngineObservation(Observation):
    """Full observable state returned after every reset() and step().

    All list fields are variable-length; their sizes are determined by the
    topology defined in ``config.yaml`` at environment initialisation time:

        inventory_levels   : length = num_suppliers + num_dcs + num_retailers
        backlog            : length = num_retailers
        demand_forecast    : length = num_retailers
        lead_times         : length = num_suppliers
        supplier_capacity  : length = num_suppliers
        active_disruptions : variable — list of disruption event dicts

    Node ordering within ``inventory_levels`` is always
    [Supplier-1…-N, DC-1…-M, Retailer-1…-K].
    """

    inventory_levels: List[float] = Field(
        default_factory=list,
        description=(
            "On-hand inventory at each node ordered as "
            "[Supplier-1…-N, DC-1…-M, Retailer-1…-K]. "
            "Length = num_suppliers + num_dcs + num_retailers."
        ),
    )
    backlog: List[float] = Field(
        default_factory=list,
        description=(
            "Unfulfilled demand (backlog) per retailer [Retailer-1…-K]. "
            "Length = num_retailers. Accumulated from previous steps."
        ),
    )
    demand_forecast: List[float] = Field(
        default_factory=list,
        description=(
            "Noisy demand forecast for the current step per retailer [Retailer-1…-K]. "
            "Length = num_retailers."
        ),
    )
    lead_times: List[float] = Field(
        default_factory=list,
        description=(
            "Current supplier lead times in steps [Supplier-1…-N]. "
            "Length = num_suppliers. May increase during disruptions."
        ),
    )
    supplier_capacity: List[float] = Field(
        default_factory=list,
        description=(
            "Available production / shipping capacity per supplier [Supplier-1…-N]. "
            "Length = num_suppliers. May decrease during disruptions."
        ),
    )
    active_disruptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of currently active disruption events. Each dict contains: "
            "type (DisruptionType value), "
            "affected_node (node name string, e.g. 'Supplier-2'), "
            "severity (float 0-1), remaining_steps (int), total_duration (int)."
        ),
    )
    service_level: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Cumulative service level — fraction of total demand fulfilled "
            "since episode start [0, 1]."
        ),
    )
    fill_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Fill rate for the most recent step [0, 1].",
    )
    total_cost: float = Field(
        default=0.0,
        description="Cumulative cost incurred since episode start.",
    )
    step_cost: float = Field(
        default=0.0,
        description="Cost incurred during the most recent step.",
    )
    in_transit_orders: int = Field(
        default=0,
        ge=0,
        description="Number of shipment orders currently in transit.",
    )


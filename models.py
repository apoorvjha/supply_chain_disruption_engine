# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Supply Chain Disruption Engine Environment.

Default network topology (configurable via config.yaml):
  - Suppliers              : node IDs 0 … num_suppliers-1
  - Distribution Centers   : node IDs num_suppliers … num_suppliers+num_dcs-1
  - Retailers              : node IDs num_suppliers+num_dcs … total_nodes-1

Flow:  Suppliers --[agent orders]--> DCs --[auto-replenish]--> Retailers --> Demand
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """Available agent action types."""

    DO_NOTHING = "do_nothing"
    """Take no action this step."""

    REORDER = "reorder"
    """Place a replenishment order from a supplier (source_node_id) to a DC
    (target_node_id).  Units enter transit and arrive after the supplier's
    current lead time."""

    EXPEDITE = "expedite"
    """Rush an order from a supplier to a DC at a cost premium; lead time is
    halved compared to a standard REORDER."""

    REROUTE = "reroute"
    """Transfer available inventory directly from source_node_id to
    target_node_id within the same tier or across tiers (e.g., DC-to-DC or
    DC-to-retailer) at a flat logistics cost."""

    ACTIVATE_BACKUP_SUPPLIER = "activate_backup_supplier"
    """Activate a contingency supplier for source_node_id. Restores ~70 % of
    baseline capacity when the primary supplier is disrupted.  One-time setup
    cost applies."""

    ADJUST_PRODUCTION = "adjust_production"
    """Temporarily raise the production capacity of source_node_id (a supplier)
    proportional to the urgency signal. Incurs a variable ramp-up cost."""

    EMERGENCY_PROCUREMENT = "emergency_procurement"
    """Buy inventory on the spot market, delivered to target_node_id in one
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

    All node IDs follow the fixed topology:
        Suppliers  → 0, 1, 2
        DCs        → 3, 4
        Retailers  → 5, 6, 7, 8
    """

    @classmethod
    def model_json_schema(cls, **kwargs):
        """Override to inline the ActionType enum so the web UI renders a dropdown."""
        schema = super().model_json_schema(**kwargs)
        defs = schema.get("$defs", {})
        action_type_def = defs.get("ActionType", {})
        props = schema.get("properties", {})
        prop = props.get("action_type", {})
        if action_type_def and "$ref" in prop and "enum" in action_type_def:
            props["action_type"] = {
                "enum": action_type_def["enum"],
                "type": action_type_def.get("type", "string"),
                "title": action_type_def.get("title", "ActionType"),
                "default": prop.get("default"),
                "description": prop.get("description", action_type_def.get("description", "")),
            }
        return schema

    action_type: ActionType = Field(
        default=ActionType.DO_NOTHING,
        description="Type of supply chain intervention to execute.",
    )
    source_node_id: int = Field(
        default=0,
        ge=0,
        description=(
            "Origin node for the action. For REORDER / EXPEDITE this must be a "
            "supplier node. For REROUTE it is the node holding excess inventory. "
            "Valid range depends on the topology configured in config.yaml."
        ),
    )
    target_node_id: int = Field(
        default=3,
        ge=0,
        description=(
            "Destination node for the action. For REORDER / EXPEDITE this must "
            "be a DC node. For REROUTE / EMERGENCY_PROCUREMENT it is the "
            "receiving node. Valid range depends on the topology configured in "
            "config.yaml."
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
    [S0…Sn, DC0…DCm, R0…Rk], matching the node IDs assigned by the config.
    """

    inventory_levels: List[float] = Field(
        default_factory=list,
        description=(
            "On-hand inventory at each node ordered as "
            "[S0…Sn, DC0…DCm, R0…Rk]. "
            "Length = num_suppliers + num_dcs + num_retailers."
        ),
    )
    backlog: List[float] = Field(
        default_factory=list,
        description=(
            "Unfulfilled demand (backlog) per retailer [R0…Rk]. "
            "Length = num_retailers. Accumulated from previous steps."
        ),
    )
    demand_forecast: List[float] = Field(
        default_factory=list,
        description=(
            "Noisy demand forecast for the current step per retailer [R0…Rk]. "
            "Length = num_retailers."
        ),
    )
    lead_times: List[float] = Field(
        default_factory=list,
        description=(
            "Current supplier lead times in steps [S0…Sn]. "
            "Length = num_suppliers. May increase during disruptions."
        ),
    )
    supplier_capacity: List[float] = Field(
        default_factory=list,
        description=(
            "Available production / shipping capacity per supplier [S0…Sn]. "
            "Length = num_suppliers. May decrease during disruptions."
        ),
    )
    active_disruptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of currently active disruption events. Each dict contains: "
            "type (DisruptionType value), affected_node (int), severity (float 0-1), "
            "remaining_steps (int), total_duration (int)."
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

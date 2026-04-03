# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Configuration loader for the Supply Chain Disruption Engine.

Reads ``config.yaml`` (or any user-supplied YAML file) and exposes a
validated ``SupplyChainConfig`` Pydantic model that the environment consumes
at initialisation time.

Usage::

    from supply_chain_disruption_engine.config import load_config

    cfg = load_config()                      # uses default config.yaml
    cfg = load_config("path/to/custom.yaml") # custom file
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, Field, model_validator

# Default config lives next to this module.
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class TopologyConfig(BaseModel):
    num_suppliers: int = Field(ge=1, description="Number of supplier nodes.")
    num_dcs: int = Field(ge=1, description="Number of distribution centre nodes.")
    num_retailers: int = Field(ge=1, description="Number of retailer nodes.")
    retailer_dc_assignment: List[int] = Field(
        description=(
            "0-based DC index that serves each retailer. "
            "Length must equal num_retailers; each value must be in [0, num_dcs)."
        )
    )

    @model_validator(mode="after")
    def _validate_assignment(self) -> "TopologyConfig":
        if len(self.retailer_dc_assignment) != self.num_retailers:
            raise ValueError(
                f"retailer_dc_assignment length ({len(self.retailer_dc_assignment)}) "
                f"must equal num_retailers ({self.num_retailers})."
            )
        for idx in self.retailer_dc_assignment:
            if idx < 0 or idx >= self.num_dcs:
                raise ValueError(
                    f"retailer_dc_assignment entry {idx} is out of range "
                    f"[0, num_dcs={self.num_dcs})."
                )
        return self


class InventoryConfig(BaseModel):
    supplier_initial: List[float] = Field(description="Initial on-hand inventory per supplier.")
    dc_initial: List[float] = Field(description="Initial on-hand inventory per DC.")
    retailer_initial: List[float] = Field(description="Initial on-hand inventory per retailer.")


class SuppliersConfig(BaseModel):
    base_lead_times: List[float] = Field(description="Replenishment lead time (steps) per supplier.")
    base_capacity: List[float] = Field(description="Max production/shipping units per step per supplier.")
    backup_capacity_fraction: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of base capacity restored by backup supplier activation.",
    )


class RetailersConfig(BaseModel):
    base_demand: List[float] = Field(description="Mean consumer demand (units/step) per retailer.")
    demand_noise_std: float = Field(ge=0.0, description="Gaussian noise std on realised demand.")
    safety_stock_days: float = Field(
        ge=0.0,
        description="DC→Retailer push target expressed as a multiple of base_demand.",
    )


class CostsConfig(BaseModel):
    holding_cost_per_unit: float = Field(ge=0.0)
    backlog_penalty_per_unit: float = Field(ge=0.0)
    order_cost_per_unit: float = Field(ge=0.0)
    expedite_premium: float = Field(ge=1.0, description="Multiplier on order_cost for EXPEDITE.")
    emergency_premium: float = Field(ge=1.0, description="Multiplier on order_cost for EMERGENCY_PROCUREMENT.")
    reroute_cost_flat: float = Field(ge=0.0)
    reroute_cost_per_unit: float = Field(ge=0.0)
    activate_backup_cost: float = Field(ge=0.0)
    production_ramp_cost_per_unit: float = Field(ge=0.0)
    production_ramp_max_fraction: float = Field(
        ge=1.0,
        description="Max supplier capacity as a multiple of its base capacity.",
    )


class DisruptionsConfig(BaseModel):
    probability_per_step: float = Field(ge=0.0, le=1.0)
    max_concurrent: int = Field(ge=0)
    min_duration: int = Field(ge=1)
    max_duration: int = Field(ge=1)
    min_severity: float = Field(ge=0.0, le=1.0)
    max_severity: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_duration_and_severity(self) -> "DisruptionsConfig":
        if self.min_duration > self.max_duration:
            raise ValueError(
                f"min_duration ({self.min_duration}) must be <= max_duration ({self.max_duration})."
            )
        if self.min_severity > self.max_severity:
            raise ValueError(
                f"min_severity ({self.min_severity}) must be <= max_severity ({self.max_severity})."
            )
        return self


class EpisodeConfig(BaseModel):
    max_steps: int = Field(ge=1, description="Number of steps before the episode terminates.")


class RewardConfig(BaseModel):
    """Weights and scaling for the normalised reward signal.

    The reward at each step is computed as:

    .. code-block:: none

        ref  = sum(initial_inventory) * holding_cost_per_unit * cost_scale_factor
        cost_efficiency = exp(-step_cost / ref)          ∈ (0, 1]

        reward = fill_rate_weight      * fill_rate
               + service_level_weight  * service_level
               + cost_efficiency_weight * cost_efficiency   ∈ [0, 1]

    All three weights must sum to 1.0.
    """

    fill_rate_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight applied to the immediate step fill rate [0, 1].",
    )
    service_level_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight applied to the cumulative service level [0, 1].",
    )
    cost_efficiency_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight applied to the cost efficiency score [0, 1].",
    )
    cost_scale_factor: float = Field(
        gt=0.0,
        description=(
            "Multiplier on the reference cost used in the exponential decay. "
            "1.0 ≡ baseline holding cost at episode start. "
            "Increase to be more forgiving of high costs."
        ),
    )

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> "RewardConfig":
        total = self.fill_rate_weight + self.service_level_weight + self.cost_efficiency_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"reward weights must sum to 1.0, got "
                f"{self.fill_rate_weight} + {self.service_level_weight} + "
                f"{self.cost_efficiency_weight} = {total:.8f}."
            )
        return self


# ---------------------------------------------------------------------------
# Root config model
# ---------------------------------------------------------------------------


class SupplyChainConfig(BaseModel):
    """Fully validated configuration for the Supply Chain Disruption Engine."""

    topology: TopologyConfig
    inventory: InventoryConfig
    suppliers: SuppliersConfig
    retailers: RetailersConfig
    costs: CostsConfig
    disruptions: DisruptionsConfig
    episode: EpisodeConfig
    reward: RewardConfig

    # ------------------------------------------------------------------
    # Cross-section length validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_lengths(self) -> "SupplyChainConfig":
        n_s = self.topology.num_suppliers
        n_d = self.topology.num_dcs
        n_r = self.topology.num_retailers

        errors: List[str] = []

        def _check(label: str, lst: List, expected: int) -> None:
            if len(lst) != expected:
                errors.append(
                    f"{label}: expected {expected} entries, got {len(lst)}."
                )

        _check("inventory.supplier_initial", self.inventory.supplier_initial, n_s)
        _check("inventory.dc_initial", self.inventory.dc_initial, n_d)
        _check("inventory.retailer_initial", self.inventory.retailer_initial, n_r)
        _check("suppliers.base_lead_times", self.suppliers.base_lead_times, n_s)
        _check("suppliers.base_capacity", self.suppliers.base_capacity, n_s)
        _check("retailers.base_demand", self.retailers.base_demand, n_r)

        if errors:
            raise ValueError("\n".join(errors))

        return self

    # ------------------------------------------------------------------
    # Derived topology helpers (computed once, used by environment)
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.topology.num_suppliers + self.topology.num_dcs + self.topology.num_retailers

    @property
    def supplier_ids(self) -> List[int]:
        return list(range(self.topology.num_suppliers))

    @property
    def dc_ids(self) -> List[int]:
        n_s = self.topology.num_suppliers
        return list(range(n_s, n_s + self.topology.num_dcs))

    @property
    def retailer_ids(self) -> List[int]:
        n_s = self.topology.num_suppliers
        n_d = self.topology.num_dcs
        return list(range(n_s + n_d, n_s + n_d + self.topology.num_retailers))

    @property
    def retailer_dc_map(self) -> List[int]:
        """Absolute DC node IDs for each retailer (same order as retailer_ids)."""
        dc_ids = self.dc_ids
        return [dc_ids[i] for i in self.topology.retailer_dc_assignment]

    @property
    def initial_inventory(self) -> List[float]:
        """Flat inventory list ordered [suppliers…, DCs…, retailers…]."""
        return (
            list(self.inventory.supplier_initial)
            + list(self.inventory.dc_initial)
            + list(self.inventory.retailer_initial)
        )


# ---------------------------------------------------------------------------
# Public loader function
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> SupplyChainConfig:
    """Load and validate a supply chain config YAML file.

    Args:
        path: Path to the YAML config file. When ``None`` the bundled
              ``config.yaml`` next to this module is used.

    Returns:
        Validated ``SupplyChainConfig`` instance.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the YAML content fails validation.
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as fh:
        raw = yaml.safe_load(fh)

    return SupplyChainConfig.model_validate(raw)

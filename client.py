# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Supply Chain Disruption Engine Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SupplyChainDisruptionEngineAction, SupplyChainDisruptionEngineObservation


class SupplyChainDisruptionEngineEnv(
    EnvClient[SupplyChainDisruptionEngineAction, SupplyChainDisruptionEngineObservation, State]
):
    """
    Client for the Supply Chain Disruption Engine Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling low-latency multi-step rollouts with full supply chain state
    returned after each action.

    Node topology:
        Suppliers  : IDs 0, 1, 2
        DCs        : IDs 3, 4
        Retailers  : IDs 5, 6, 7, 8

    Example:
        >>> with SupplyChainDisruptionEngineEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(obs.inventory_levels)   # [600.0, 500.0, 450.0, 220.0, ...]
        ...
        ...     action = SupplyChainDisruptionEngineAction(
        ...         action_type="reorder",
        ...         source_node_id=0,   # Supplier S0
        ...         target_node_id=3,   # DC0
        ...         quantity=200.0,
        ...     )
        ...     result = env.step(action)
        ...     print(result.observation.fill_rate)
        ...     print(result.observation.active_disruptions)

    Example with Docker:
        >>> client = SupplyChainDisruptionEngineEnv.from_docker_image(
        ...     "supply_chain_disruption_engine-env:latest"
        ... )
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(
        ...         SupplyChainDisruptionEngineAction(action_type="do_nothing")
        ...     )
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SupplyChainDisruptionEngineAction) -> Dict:
        """Convert a SupplyChainDisruptionEngineAction to a JSON-serialisable dict.

        Args:
            action: The action chosen by the agent.

        Returns:
            Dict suitable for transmission as the step request body.
        """
        return {
            "action_type": action.action_type.value,
            "source_node": action.source_node,
            "target_node": action.target_node,
            "quantity": action.quantity,
            "urgency": action.urgency,
            "metadata": action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupplyChainDisruptionEngineObservation]:
        """Parse the server's JSON response into a typed StepResult.

        Args:
            payload: Raw JSON dict returned by the environment server.

        Returns:
            StepResult wrapping a SupplyChainDisruptionEngineObservation.
        """
        obs_data = payload.get("observation", {})

        observation = SupplyChainDisruptionEngineObservation(
            inventory_levels=obs_data.get("inventory_levels", []),
            backlog=obs_data.get("backlog", []),
            demand_forecast=obs_data.get("demand_forecast", []),
            lead_times=obs_data.get("lead_times", []),
            supplier_capacity=obs_data.get("supplier_capacity", []),
            active_disruptions=obs_data.get("active_disruptions", []),
            service_level=obs_data.get("service_level", 1.0),
            fill_rate=obs_data.get("fill_rate", 1.0),
            total_cost=obs_data.get("total_cost", 0.0),
            step_cost=obs_data.get("step_cost", 0.0),
            in_transit_orders=obs_data.get("in_transit_orders", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse the server's state response into a State object.

        Args:
            payload: JSON response data from the /state endpoint.

        Returns:
            State object carrying episode_id and step_count.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

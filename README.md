---
title: Supply Chain Disruption Engine Environment Server
emoji: 🏭
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Supply Chain Disruption Engine

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment for **multi-echelon supply chain disruption management**. An agent learns to maintain service levels and minimise costs while responding to stochastic disruptions (natural disasters, labour strikes, demand spikes, supplier bankruptcies, and more).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Network Topology](#network-topology)
4. [Observations](#observations)
5. [Actions](#actions)
6. [Disruption Events](#disruption-events)
7. [Reward Signal](#reward-signal)
8. [Episode Lifecycle](#episode-lifecycle)
9. [Configuration Reference](#configuration-reference)
10. [Quick Start](#quick-start)
11. [Advanced Usage](#advanced-usage)
12. [Project Structure](#project-structure)
13. [Deployment](#deployment)

---

## Overview

| Property | Value |
|---|---|
| Environment type | Multi-echelon supply chain (3-tier) |
| Default topology | 3 Suppliers → 2 DCs → 4 Retailers (9 nodes) |
| Topology | **Fully configurable** via `config.yaml` |
| Action space | 7 discrete intervention types + continuous quantity/urgency |
| Observation space | 12 fields; list sizes scale with topology |
| Reward range | **[0, 1]** — weighted sum of fill rate, service level, cost efficiency |
| Episode length | Configurable (default 30 steps) |
| Disruptions | 6 types, stochastic spawn and recovery |
| Concurrency | Concurrent WebSocket sessions supported |
| Transport | HTTP REST + persistent WebSocket |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenEnv Client                          │
│  SupplyChainDisruptionEngineEnv  (client.py)                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  reset() ──► POST /reset  │  step() ──► WS /ws     │    │
│  │  _step_payload()          │  _parse_result()        │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP / WebSocket
┌────────────────────────▼────────────────────────────────────┐
│                   FastAPI Server  (server/app.py)           │
│  POST /reset   POST /step   GET /state   GET /schema        │
│  WS /ws        GET /health  GET /web (interactive UI)       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│        SupplyChainDisruptionEngineEnvironment                │
│           (server/supply_chain_disruption_engine_           │
│                    environment.py)                          │
│                                                             │
│  reset(seed, episode_id)  step(action)  state (property)   │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │              SupplyChainConfig                      │    │
│  │  (loaded from config.yaml at __init__ time)         │    │
│  │  topology · inventory · suppliers · retailers       │    │
│  │  costs · disruptions · episode · reward             │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Key components

| File | Responsibility |
|---|---|
| `models.py` | Pydantic `Action` and `Observation` data models; `ActionType` and `DisruptionType` enums |
| `config.py` | `SupplyChainConfig` Pydantic model; `load_config()` loader; cross-field validation |
| `config.yaml` | Single source of truth for all tunable parameters |
| `server/supply_chain_disruption_engine_environment.py` | Core simulation: state transitions, material flow, disruptions, reward |
| `server/app.py` | FastAPI application; OpenEnv HTTP + WebSocket endpoints |
| `client.py` | `SupplyChainDisruptionEngineEnv` — typed Python client over WebSocket |

---

## Network Topology

The supply chain is modelled as a **directed three-tier network**. Node IDs are assigned contiguously:

```
ID range                    Tier
────────────────────────────────────────────────────────────────
0 … num_suppliers-1         Suppliers   (production / raw material sources)
num_suppliers … +num_dcs-1  Distribution Centres  (intermediate warehouses)
+num_dcs … total_nodes-1    Retailers   (consumer-facing demand points)
```

**Default topology** (`num_suppliers=3, num_dcs=2, num_retailers=4`):

```
  S0 (0) ──┐
  S1 (1) ──┼──► DC0 (3) ──► R0 (5)
  S2 (2) ──┤             └──► R1 (6)
            └──► DC1 (4) ──► R2 (7)
                          └──► R3 (8)
```

Material flow direction: **Suppliers → DCs → Retailers → Consumer demand**

- The agent places orders from Suppliers to DCs. Shipments enter a transit queue and arrive after the supplier's current lead time.
- DCs automatically push inventory to their assigned retailers each step to maintain a configurable safety stock level.
- Consumer demand is realised stochastically at each retailer every step.

The topology (number of nodes per tier and DC-to-retailer assignment) is **fully configurable** in `config.yaml`. The environment, observation, and client all adapt automatically to any topology.

---

## Observations

`SupplyChainDisruptionEngineObservation` is returned by both `reset()` and `step()`. All list fields scale with the configured topology.

| Field | Type | Length | Description |
|---|---|---|---|
| `inventory_levels` | `List[float]` | `num_nodes` | On-hand inventory at every node ordered `[S0…Sn, DC0…DCm, R0…Rk]` |
| `backlog` | `List[float]` | `num_retailers` | Unfulfilled demand accumulated per retailer |
| `demand_forecast` | `List[float]` | `num_retailers` | Noisy 1-step-ahead demand forecast per retailer |
| `lead_times` | `List[float]` | `num_suppliers` | Current replenishment lead time per supplier (steps); rises during disruptions |
| `supplier_capacity` | `List[float]` | `num_suppliers` | Current max production/shipping units per step per supplier; drops during disruptions |
| `active_disruptions` | `List[dict]` | variable | Each dict: `type`, `affected_node`, `severity`, `remaining_steps`, `total_duration` |
| `service_level` | `float` | — | Cumulative fraction of total demand fulfilled since episode start ∈ [0, 1] |
| `fill_rate` | `float` | — | Fraction of this step's demand fulfilled ∈ [0, 1] |
| `total_cost` | `float` | — | Cumulative cost incurred since episode start ($) |
| `step_cost` | `float` | — | Cost incurred during this step ($) |
| `in_transit_orders` | `int` | — | Number of shipments currently in the transit queue |
| `reward` | `float` | — | Step reward ∈ [0, 1] |
| `done` | `bool` | — | `True` when `step_count >= max_steps` |
| `metadata` | `dict` | — | `step`, `episode_id`, `backup_active`, `topology` |

---

## Actions

`SupplyChainDisruptionEngineAction` fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `action_type` | `ActionType` | `do_nothing` | Intervention to execute (see table below) |
| `source_node_id` | `int ≥ 0` | `0` | Origin node |
| `target_node_id` | `int ≥ 0` | `3` | Destination node |
| `quantity` | `float [0, 2000]` | `0.0` | Units to order / transfer / procure |
| `urgency` | `float [0, 1]` | `0.5` | Ramp-up urgency for `ADJUST_PRODUCTION` |

### Action types

| `action_type` | `source_node_id` | `target_node_id` | Effect | Cost |
|---|---|---|---|---|
| `do_nothing` | ignored | ignored | No-op | $0 |
| `reorder` | Supplier | DC | Places order into transit queue; arrives after lead time | `quantity × order_cost_per_unit` |
| `expedite` | Supplier | DC | Same as REORDER but lead time is halved | `quantity × order_cost_per_unit × expedite_premium` |
| `reroute` | Any node | Any node | Transfers inventory directly (no lead time) | `reroute_cost_flat + quantity × reroute_cost_per_unit` |
| `activate_backup_supplier` | Supplier | ignored | Restores supplier capacity to `backup_capacity_fraction × base`. One-time per supplier per episode | `activate_backup_cost` |
| `adjust_production` | Supplier | ignored | Ramps up capacity: `min(base × max_fraction, current × (1 + urgency × (max_fraction−1)))` | `delta_capacity × production_ramp_cost_per_unit` |
| `emergency_procurement` | ignored | Any node | Purchases from spot market; arrives in exactly 1 step | `quantity × order_cost_per_unit × emergency_premium` |

---

## Disruption Events

Each step, a new disruption spawns with probability `disruptions.probability_per_step` (if below the concurrent cap). Duration and severity are sampled uniformly within configured ranges. Once expired, nodes recover automatically.

| `DisruptionType` | Affected tier | Effect during disruption | Recovery |
|---|---|---|---|
| `natural_disaster` | Any | Destroys `severity` fraction of node inventory; reduces supplier capacity by the same fraction | Full inventory remains (can't undo destruction); capacity restored |
| `transportation_failure` | Supplier | Inflates lead time by `×(1 + severity×2.5)` | Lead time reset to base |
| `labor_strike` | Supplier | Cuts capacity by `severity × 85%` | Capacity restored (or to backup fraction if backup is active) |
| `demand_spike` | Retailer | Multiplies base demand by `(1 + severity)` | Base demand divided back down |
| `supplier_bankruptcy` | Supplier | Sets capacity to 0; requires `ACTIVATE_BACKUP_SUPPLIER` to recover | Capacity restored at expiry |
| `port_congestion` | Supplier | Same lead-time inflation as `transportation_failure` | Lead time reset to base |

Each disruption dict in the observation contains:
```json
{
  "type": "labor_strike",
  "affected_node": 1,
  "severity": 0.62,
  "remaining_steps": 3,
  "total_duration": 5
}
```

---

## Reward Signal

The reward is bounded to **[0, 1]** and computed every step as a weighted sum of three normalised components:

$$\text{ref} = \left(\sum_i \text{inventory}_i^{(0)}\right) \times c_{\text{hold}} \times \lambda$$

$$\text{cost\_efficiency} = e^{-\,\text{step\_cost}\;/\;\text{ref}} \in (0,\,1]$$

$$\boxed{\text{reward} = w_f \cdot \text{fill\_rate} + w_s \cdot \text{service\_level} + w_c \cdot \text{cost\_efficiency}}$$

| Component | Symbol | Measures | Range |
|---|---|---|---|
| Fill rate | $w_f \cdot \text{fill\_rate}$ | Fraction of **this step's** demand fulfilled | [0, 1] |
| Service level | $w_s \cdot \text{service\_level}$ | Cumulative fraction of **all demand** fulfilled since reset | [0, 1] |
| Cost efficiency | $w_c \cdot e^{-\text{step\_cost}/\text{ref}}$ | How cheaply the step was executed (higher = cheaper) | (0, 1] |

Where:
- $\text{ref}$ is the reference cost — baseline holding cost at initial inventory with no backlog or actions. `cost_scale_factor` (λ) tunes how aggressively cost is penalised.
- Weights $w_f + w_s + w_c = 1.0$ (validated at config load time).
- Default weights: fill rate 0.40, service level 0.40, cost efficiency 0.20.

**Intuition**: A well-run supply chain that fulfils all demand cheaply scores near 1.0. Stockouts and backlog drag fill rate down; excessive ordering, expediting, or emergency procurement drag cost efficiency down; sustained backlog accumulation drags service level down.

---

## Episode Lifecycle

```
reset(seed?, episode_id?)
    │
    ├─ Initialises inventory, lead times, capacities from config
    ├─ Clears transit queue, disruptions, backlog, accumulators
    └─ Returns initial Observation (step_count=0, reward=0)

for each step:
    step(action)
        │
        ├─ 1. Apply agent action  →  action_cost
        ├─ 2. Tick disruptions   →  spawn/expire/recover
        ├─ 3. Deliver orders     →  countdown transit queue; credit arrived stock
        ├─ 4. DC→Retailer push   →  fill gap to safety stock
        ├─ 5. Realise demand     →  stochastic consumer demand; update backlog
        ├─ 6. Compute costs      →  holding + backlog + action_cost
        ├─ 7. Compute reward     →  [0, 1] weighted signal
        └─ Returns Observation   →  done=True when step_count == max_steps
```

---

## Configuration Reference

All parameters live in `supply_chain_disruption_engine/config.yaml`. The server must be restarted after changes.

### `topology`

| Key | Type | Description |
|---|---|---|
| `num_suppliers` | int ≥ 1 | Number of supplier nodes |
| `num_dcs` | int ≥ 1 | Number of distribution centre nodes |
| `num_retailers` | int ≥ 1 | Number of retailer nodes |
| `retailer_dc_assignment` | List[int] | 0-based DC index serving each retailer. Length must equal `num_retailers`; each value in `[0, num_dcs)` |

### `inventory`

| Key | Type | Length | Description |
|---|---|---|---|
| `supplier_initial` | List[float] | `num_suppliers` | Starting on-hand inventory per supplier |
| `dc_initial` | List[float] | `num_dcs` | Starting on-hand inventory per DC |
| `retailer_initial` | List[float] | `num_retailers` | Starting on-hand inventory per retailer |

### `suppliers`

| Key | Type | Length | Description |
|---|---|---|---|
| `base_lead_times` | List[float] | `num_suppliers` | Replenishment lead time in steps |
| `base_capacity` | List[float] | `num_suppliers` | Max units shipped per step |
| `backup_capacity_fraction` | float [0,1] | — | Fraction of base capacity restored by backup activation |

### `retailers`

| Key | Type | Length | Description |
|---|---|---|---|
| `base_demand` | List[float] | `num_retailers` | Mean consumer demand per step |
| `demand_noise_std` | float ≥ 0 | — | Gaussian noise std on realised demand |
| `safety_stock_days` | float ≥ 0 | — | DC→Retailer push target as a multiplier of `base_demand` |

### `costs`

| Key | Description |
|---|---|
| `holding_cost_per_unit` | $/unit/step across all nodes |
| `backlog_penalty_per_unit` | $/unfulfilled unit/step |
| `order_cost_per_unit` | $/unit for REORDER |
| `expedite_premium` | Multiplier on `order_cost_per_unit` for EXPEDITE |
| `emergency_premium` | Multiplier on `order_cost_per_unit` for EMERGENCY_PROCUREMENT |
| `reroute_cost_flat` | Fixed $ per REROUTE action |
| `reroute_cost_per_unit` | $/unit transferred for REROUTE |
| `activate_backup_cost` | One-time $ for ACTIVATE_BACKUP_SUPPLIER |
| `production_ramp_cost_per_unit` | $/unit of capacity increase for ADJUST_PRODUCTION |
| `production_ramp_max_fraction` | Max supplier capacity as multiple of base (≥ 1.0) |

### `disruptions`

| Key | Description |
|---|---|
| `probability_per_step` | P(new disruption spawns each step) ∈ [0, 1] |
| `max_concurrent` | Maximum simultaneous active disruptions |
| `min_duration` / `max_duration` | Disruption duration range (steps) |
| `min_severity` / `max_severity` | Disruption severity range ∈ [0, 1] |

### `episode`

| Key | Description |
|---|---|
| `max_steps` | Steps per episode; `done=True` when reached |

### `reward`

| Key | Description |
|---|---|
| `fill_rate_weight` | Weight on immediate fill rate (default 0.40) |
| `service_level_weight` | Weight on cumulative service level (default 0.40) |
| `cost_efficiency_weight` | Weight on cost efficiency score (default 0.20) |
| `cost_scale_factor` | Reference cost multiplier λ. Increase to forgive higher costs (default 1.0) |

> **Constraint**: `fill_rate_weight + service_level_weight + cost_efficiency_weight` must equal exactly `1.0`. Config loading raises `ValidationError` otherwise.

---

## Quick Start

### 1. Using the Python client (server already running)

```python
from supply_chain_disruption_engine import (
    SupplyChainDisruptionEngineAction,
    SupplyChainDisruptionEngineEnv,
)
from supply_chain_disruption_engine.models import ActionType

with SupplyChainDisruptionEngineEnv(base_url="http://localhost:8000") as env:
    # Reset — returns initial observation
    result = env.reset()
    obs = result.observation
    print("Inventory levels:", obs.inventory_levels)
    print("Active disruptions:", obs.active_disruptions)

    # Step 1: reorder from Supplier 0 to DC 0
    action = SupplyChainDisruptionEngineAction(
        action_type=ActionType.REORDER,
        source_node_id=0,   # Supplier S0
        target_node_id=3,   # DC DC0
        quantity=200.0,
    )
    result = env.step(action)
    print(f"Reward: {result.observation.reward:.4f}")
    print(f"Fill rate: {result.observation.fill_rate:.4f}")
    print(f"Service level: {result.observation.service_level:.4f}")
    print(f"In transit: {result.observation.in_transit_orders}")

    # Step 2: expedite from Supplier 1 to DC 1 (half lead time, 3× cost)
    result = env.step(SupplyChainDisruptionEngineAction(
        action_type=ActionType.EXPEDITE,
        source_node_id=1,
        target_node_id=4,
        quantity=150.0,
    ))

    # Step 3: activate backup supplier after a bankruptcy disruption
    result = env.step(SupplyChainDisruptionEngineAction(
        action_type=ActionType.ACTIVATE_BACKUP_SUPPLIER,
        source_node_id=2,
    ))

    print(f"Done: {result.observation.done}")
    print(f"Total cost: ${result.observation.total_cost:,.2f}")
```

### 2. Using Docker

```bash
# Build the image
docker build -t supply_chain_disruption_engine-env:latest -f server/Dockerfile .

# Run with automatic Docker management
from supply_chain_disruption_engine import SupplyChainDisruptionEngineEnv, SupplyChainDisruptionEngineAction
from supply_chain_disruption_engine.models import ActionType

client = SupplyChainDisruptionEngineEnv.from_docker_image("supply_chain_disruption_engine-env:latest")
try:
    result = client.reset()
    result = client.step(SupplyChainDisruptionEngineAction(action_type=ActionType.DO_NOTHING))
finally:
    client.close()
```

### 3. Running the server locally

```bash
# Development (auto-reload)
ENABLE_WEB_INTERFACE=true uvicorn supply_chain_disruption_engine.server.app:app --host 0.0.0.0 --port 8000

# Or via the project entry-point
uv run --project . server

# Or directly
python -m supply_chain_disruption_engine.server.app
```

### 4. Custom topology

Create a custom YAML file and pass it at construction:

```python
from supply_chain_disruption_engine.server.supply_chain_disruption_engine_environment import (
    SupplyChainDisruptionEngineEnvironment,
)

# 2 suppliers, 3 DCs, 6 retailers
env = SupplyChainDisruptionEngineEnvironment(config_path="my_config.yaml")
obs = env.reset(seed=42)
print(f"Nodes: {len(obs.inventory_levels)}")   # → 11
```

Minimum valid `my_config.yaml` for a 2S/3DC/6R network — just copy `config.yaml` and change the topology and list lengths to match.

---

## Advanced Usage

### Running a full RL episode

```python
from supply_chain_disruption_engine import SupplyChainDisruptionEngineEnv, SupplyChainDisruptionEngineAction
from supply_chain_disruption_engine.models import ActionType

with SupplyChainDisruptionEngineEnv(base_url="http://localhost:8000") as env:
    result = env.reset(seed=0)
    total_reward = 0.0

    while not result.observation.done:
        obs = result.observation

        # Simple heuristic: reorder if any DC is below threshold
        action = SupplyChainDisruptionEngineAction(action_type=ActionType.DO_NOTHING)
        dc_inventory = obs.inventory_levels[3]   # DC0 for default topology
        if dc_inventory < 100.0:
            action = SupplyChainDisruptionEngineAction(
                action_type=ActionType.REORDER,
                source_node_id=0,
                target_node_id=3,
                quantity=300.0,
            )

        result = env.step(action)
        total_reward += result.observation.reward

    print(f"Episode reward: {total_reward:.3f}")
    print(f"Final service level: {result.observation.service_level:.4f}")
    print(f"Total cost: ${result.observation.total_cost:,.2f}")
```

### Concurrent WebSocket sessions

The environment supports multiple simultaneous sessions (`SUPPORTS_CONCURRENT_SESSIONS = True`):

```python
from concurrent.futures import ThreadPoolExecutor
from supply_chain_disruption_engine import SupplyChainDisruptionEngineEnv, SupplyChainDisruptionEngineAction
from supply_chain_disruption_engine.models import ActionType

def run_episode(seed: int) -> float:
    with SupplyChainDisruptionEngineEnv(base_url="http://localhost:8000") as env:
        result = env.reset(seed=seed)
        total = 0.0
        while not result.observation.done:
            result = env.step(SupplyChainDisruptionEngineAction(
                action_type=ActionType.REORDER,
                source_node_id=0, target_node_id=3, quantity=150.0,
            ))
            total += result.observation.reward
        return total

with ThreadPoolExecutor(max_workers=4) as pool:
    rewards = list(pool.map(run_episode, range(4)))
print("Episode rewards:", rewards)
```

To increase the concurrent session limit, edit `server/app.py`:

```python
app = create_app(
    SupplyChainDisruptionEngineEnvironment,
    SupplyChainDisruptionEngineAction,
    SupplyChainDisruptionEngineObservation,
    max_concurrent_envs=8,   # up from 1
)
```

### Testing the environment directly (no server)

```python
from supply_chain_disruption_engine.server.supply_chain_disruption_engine_environment import (
    SupplyChainDisruptionEngineEnvironment,
)
from supply_chain_disruption_engine.models import ActionType, SupplyChainDisruptionEngineAction

env = SupplyChainDisruptionEngineEnvironment()
obs = env.reset(seed=42)

for _ in range(30):
    obs = env.step(SupplyChainDisruptionEngineAction(
        action_type=ActionType.REORDER, source_node_id=0, target_node_id=3, quantity=150.0
    ))

print("Done:", obs.done)
print("Service level:", obs.service_level)
print("Total cost:", obs.total_cost)
print("Episode ID:", env.state.episode_id)
```

---

## Project Structure

```
supply_chain_disruption_engine/
├── __init__.py                          # Package exports
├── client.py                            # SupplyChainDisruptionEngineEnv (WebSocket client)
├── models.py                            # Action / Observation Pydantic models;
│                                        # ActionType and DisruptionType enums
├── config.py                            # SupplyChainConfig Pydantic model;
│                                        # load_config() loader + cross-field validation
├── config.yaml                          # All tunable parameters (topology, costs,
│                                        # disruptions, reward weights, episode length)
├── openenv.yaml                         # OpenEnv manifest (name, runtime, port)
├── pyproject.toml                       # Project metadata and dependencies
├── README.md                            # This file
└── server/
    ├── __init__.py                      # Server package exports
    ├── app.py                           # FastAPI app (HTTP + WebSocket endpoints)
    ├── supply_chain_disruption_engine_  # Core environment simulation:
    │   environment.py                   # reset / step / state; material flow;
    │                                    # disruption dynamics; reward computation
    ├── requirements.txt                 # Server-only dependencies
    └── Dockerfile                       # Container image definition
```

---

## Deployment

### Hugging Face Spaces

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# To a specific repo
openenv push --repo-id my-org/supply-chain-env

# As a private space
openenv push --private
```

After deployment the space exposes:

| Endpoint | Description |
|---|---|
| `GET  /web` | Interactive web UI for exploring the environment |
| `POST /reset` | Reset the environment |
| `POST /step` | Execute an action |
| `GET  /state` | Current episode state |
| `GET  /schema` | Action / Observation JSON schemas |
| `WS   /ws` | Persistent WebSocket session |
| `GET  /docs` | OpenAPI / Swagger documentation |
| `GET  /health` | Health check |



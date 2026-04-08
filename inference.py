"""
Inference Script — Supply Chain Disruption Engine
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    IMAGE_NAME          The Docker image name for the environment (from_docker_image).
- Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1].
  Example:
    [START] task=supply_chain_disruption env=supply_chain_disruption_engine model=Qwen2.5-72B
    [STEP] step=1 action=reorder(src=Supplier-1,tgt=DC-1,qty=200,urg=0.50) reward=0.82 done=false error=null
    [STEP] step=2 action=do_nothing(src=Supplier-1,tgt=DC-1,qty=0,urg=0.00) reward=0.85 done=false error=null
    [END] success=true steps=30 score=0.840 rewards=0.82,0.85,...
AGENT STRATEGY
  The LLM agent receives the full supply chain state each step and outputs a
  JSON action.  If the LLM response cannot be parsed, a proven heuristic policy
  is used as fallback.
  Priority order (highest → lowest):
    1. EMERGENCY_PROCUREMENT — DC stock < 0.5-day demand AND active retailer backlog
    2. EXPEDITE              — DC stock < full lead-time coverage (avoidable stockout)
    3. ACTIVATE_BACKUP       — supplier capacity ≤ 20 % of baseline (disruption)
    4. REORDER               — DC stock < (lead_time + 1.5 safety) × demand rate
    5. ADJUST_PRODUCTION     — supplier disrupted, DC approaching reorder point
    6. REROUTE               — balanced inventory across DCs (donor >2× avg DoS)
    7. DO_NOTHING            — supply comfortable at all nodes
"""

import json
import math
import os
import re
import textwrap
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional
import asyncio

from openai import OpenAI

from supply_chain_disruption_engine import SupplyChainDisruptionEngineEnv, ActionType, SupplyChainDisruptionEngineAction, SupplyChainDisruptionEngineObservation
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present
# ── Environment / model configuration ────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")  # Docker image for the environment
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")  # API
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")  # fallback env image URL
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("SUPPLY_CHAIN_TASK", "all")
BENCHMARK = os.getenv("SUPPLY_CHAIN_BENCHMARK", "supply_chain_disruption_engine")

# ── Episode / model settings ──────────────────────────────────────────────────

MAX_STEPS = 30           # must match episode.max_steps in config.yaml
TEMPERATURE = 0.2        # low temperature → more deterministic decisions
MAX_TOKENS = 256         # enough for one JSON action object
SUCCESS_SCORE_THRESHOLD = 0.5   # mean per-step reward ≥ 0.5 counts as success

# ── Topology constants (mirror config.yaml — update here if YAML changes) ─────

NUM_SUPPLIERS = 3
NUM_DCS = 3
NUM_RETAILERS = 4

SUPPLIER_LEAD_TIMES: List[float] = [2.0, 3.0, 4.0]       # steps
SUPPLIER_BASE_CAPACITY: List[float] = [500.0, 420.0, 360.0]  # units/step
BACKUP_CAPACITY_FRACTION: float = 0.70

BASE_DEMAND: List[float] = [60.0, 55.0, 65.0, 50.0]   # units/step per retailer
SAFETY_STOCK_DAYS: float = 1.5

# DC (0-based) → list of retailer indices it serves
DC_RETAILER_MAP: Dict[int, List[int]] = {0: [0, 1], 1: [2], 2: [3]}
# DC (0-based) → preferred supplier index (shortest lead time / highest cap)
DC_PRIMARY_SUPPLIER: Dict[int, int] = {0: 0, 1: 1, 2: 2}

# Reference cost for cost-efficiency calculation (mirrors env reward formula):
#   reference_cost = sum(all initial inventory) × holding_cost_per_unit × cost_scale_factor
#   = (600+500+450 + 220+200+180 + 90+80+95+75) × 0.50 × 1.0  =  1245.0
REFERENCE_COST: float = (
    sum([600.0, 500.0, 450.0])          # supplier initial inventory
    + sum([220.0, 200.0, 180.0])        # DC initial inventory
    + sum([90.0, 80.0, 95.0, 75.0])    # retailer initial inventory
) * 0.50 * 1.0   # holding_cost_per_unit × cost_scale_factor

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert supply chain manager optimising a multi-echelon distribution
    network over a 30-step episode.  Your goal is to maximise the reward:
        reward = 0.40 × fill_rate + 0.40 × service_level + 0.20 × cost_efficiency
    NETWORK TOPOLOGY (Suppliers → DCs → Retailers):
      Supplier-1  lead=2 steps  cap≈500/step  →  DC-1 (primary)
      Supplier-2  lead=3 steps  cap≈420/step  →  DC-2 (primary)
      Supplier-3  lead=4 steps  cap≈360/step  →  DC-3 (primary)
      DC-1 serves Retailer-1 (≈60/step) and Retailer-2 (≈55/step) → 115/step total
      DC-2 serves Retailer-3 (≈65/step)
      DC-3 serves Retailer-4 (≈50/step)
    COST STRUCTURE (backlog is very expensive — avoid stockouts!):
      Holding cost   : $0.50/unit/step (all nodes)
      Backlog penalty: $8.00/unit/step  ← 16× more costly than holding
      Reorder        : $2.00/unit  (standard lead time)
      Expedite       : $6.00/unit  (half lead time — use when DC is running low)
      Emergency      : $14.00/unit (1-step delivery — last resort)
      Reroute        : $50 flat + $0.50/unit (instant DC-to-DC or DC-to-retailer)
      Activate backup: $250 one-time (restores 70 % of supplier capacity)
    DECISION PRIORITY (assess in this order each step):
      1. emergency_procurement — DC stock < 0.5 × day demand AND active backlog
      2. expedite              — DC stock < (lead_time × demand_rate) [stockout soon]
      3. activate_backup_supplier — supplier capacity ≤ 20 % of baseline
      4. reorder               — DC stock < (lead_time + 1.5 safety) × demand_rate
      5. adjust_production     — supplier disrupted, DC approaching reorder point
      6. reroute               — a DC has >2× avg days-of-supply vs another <0.5×
      7. do_nothing            — supply comfortable at all nodes
    AVAILABLE ACTIONS:
      do_nothing            — no intervention this step
      reorder               — source=Supplier-X, target=DC-Y, quantity=Q
      expedite              — source=Supplier-X, target=DC-Y, quantity=Q
      reroute               — source=Node-A, target=Node-B, quantity=Q
      activate_backup_supplier — source=Supplier-X  (target ignored, quantity ignored)
      adjust_production     — source=Supplier-X, urgency∈[0,1]
      emergency_procurement — target=DC-Y, quantity=Q  (source ignored)
    OUTPUT FORMAT: Reply with ONLY a single JSON object — no markdown fences, no
    explanation.  All five fields are required.  Examples:
      {"action_type":"reorder","source_node":"Supplier-1","target_node":"DC-2","quantity":180.0,"urgency":0.5}
      {"action_type":"emergency_procurement","source_node":"Supplier-1","target_node":"DC-1","quantity":120.0,"urgency":1.0}
      {"action_type":"do_nothing","source_node":"Supplier-1","target_node":"DC-1","quantity":0.0,"urgency":0.0}
    """
).strip()

# ── Logging helpers ───────────────────────────────────────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation formatting ─────────────────────────────────────────────────────


def _dc_demand_rate(dc_idx: int) -> float:
    """Total consumer demand rate (units/step) served by a given DC."""
    return sum(BASE_DEMAND[r] for r in DC_RETAILER_MAP[dc_idx])


def _days_of_supply(inventory: float, demand_rate: float) -> float:
    """Steps of supply remaining at the current demand rate."""
    return inventory / demand_rate if demand_rate > 0 else float("inf")


def format_observation(
    obs: SupplyChainDisruptionEngineObservation, step: int
) -> str:
    """Format the observation into a concise, LLM-readable state summary."""
    inv = obs.inventory_levels
    sup_inv = inv[:NUM_SUPPLIERS]
    dc_inv = inv[NUM_SUPPLIERS : NUM_SUPPLIERS + NUM_DCS]
    ret_inv = inv[NUM_SUPPLIERS + NUM_DCS :]

    dc_dos = [
        _days_of_supply(dc_inv[i], _dc_demand_rate(i)) for i in range(NUM_DCS)
    ]

    # Format active disruptions
    if obs.active_disruptions:
        dis_parts = [
            (
                f"{d['type']}@"
                f"{d.get('affected_node_name', d.get('affected_node', '?'))} "
                f"sev={d['severity']:.2f} rem={d['remaining_steps']}steps"
            )
            for d in obs.active_disruptions
        ]
        dis_text = "; ".join(dis_parts)
    else:
        dis_text = "None"

    return textwrap.dedent(
        f"""
        STEP {step}/{MAX_STEPS}
        Inventory suppliers : {', '.join(f'Supplier-{i+1}={v:.0f}' for i, v in enumerate(sup_inv))}
        Inventory DCs       : {', '.join(f'DC-{i+1}={v:.0f}(DoS={dc_dos[i]:.1f}steps)' for i, v in enumerate(dc_inv))}
        Inventory retailers : {', '.join(f'Retailer-{i+1}={v:.0f}' for i, v in enumerate(ret_inv))}
        Backlog (retailers) : {', '.join(f'Retailer-{i+1}={v:.0f}' for i, v in enumerate(obs.backlog))}
        Demand forecast     : {', '.join(f'Retailer-{i+1}={v:.0f}' for i, v in enumerate(obs.demand_forecast))}
        Lead times          : {', '.join(f'Supplier-{i+1}={v:.1f}steps' for i, v in enumerate(obs.lead_times))}
        Supplier capacity   : {', '.join(f'Supplier-{i+1}={v:.0f}' for i, v in enumerate(obs.supplier_capacity))}
        In-transit orders   : {obs.in_transit_orders}
        Active disruptions  : {dis_text}
        Fill rate (step)    : {obs.fill_rate:.3f}
        Service level (cum) : {obs.service_level:.3f}
        Step cost           : ${obs.step_cost:.2f}
        Total cost          : ${obs.total_cost:.2f}
        """
    ).strip()


# ── Heuristic fallback policy ──────────────────────────────────────────────────


def _node_name(tier: str, one_based: int) -> str:
    return f"{tier}-{one_based}"


def _pick_best_supplier(
    dc_idx: int,
    cap: List[float],
    lead_times: List[float],
    disruptions: list,
) -> int:
    """Return the best available supplier index for a given DC.
    Prefers the primary (lowest lead-time) supplier; falls back to any
    supplier with available capacity if the primary is disrupted.
    """
    disrupted_names = {d.get("affected_node_name", "") for d in disruptions}
    primary = DC_PRIMARY_SUPPLIER[dc_idx]
    if _node_name("Supplier", primary + 1) not in disrupted_names and cap[primary] > 0:
        return primary
    # Fall back: shortest lead time with available capacity
    candidates = sorted(range(NUM_SUPPLIERS), key=lambda s: (lead_times[s], -cap[s]))
    for s in candidates:
        if cap[s] > 0:
            return s
    return primary  # last resort even if disrupted


def heuristic_action(
    obs: SupplyChainDisruptionEngineObservation,
) -> SupplyChainDisruptionEngineAction:
    """Rule-based policy derived from supply chain domain knowledge.
    Evaluates seven intervention types in descending priority and returns
    the first action whose trigger condition is met.
    """
    inv = obs.inventory_levels
    dc_inv = [inv[NUM_SUPPLIERS + i] for i in range(NUM_DCS)]
    lead_times = list(obs.lead_times)
    cap = list(obs.supplier_capacity)
    disruptions = obs.active_disruptions or []

    # ── 1. Emergency procurement ─────────────────────────────────────────
    for dc_idx in range(NUM_DCS):
        demand_rate = _dc_demand_rate(dc_idx)
        dc_backlog = sum(obs.backlog[r] for r in DC_RETAILER_MAP[dc_idx])
        if dc_inv[dc_idx] <= demand_rate * 0.5 and dc_backlog > 0:
            qty = demand_rate * 2  # buy two steps of demand immediately
            dc_name = _node_name("DC", dc_idx + 1)
            return SupplyChainDisruptionEngineAction(
                action_type=ActionType.EMERGENCY_PROCUREMENT,
                source_node=dc_name,  # ignored by the environment
                target_node=dc_name,
                quantity=qty,
                urgency=1.0,
            )

    # ── 2. Expedite ──────────────────────────────────────────────────────
    for dc_idx in range(NUM_DCS):
        sup_idx = DC_PRIMARY_SUPPLIER[dc_idx]
        lead = lead_times[sup_idx]
        demand_rate = _dc_demand_rate(dc_idx)
        if dc_inv[dc_idx] < lead * demand_rate:
            chosen_sup = _pick_best_supplier(dc_idx, cap, lead_times, disruptions)
            reorder_lvl = (lead_times[chosen_sup] + SAFETY_STOCK_DAYS) * demand_rate
            qty = min(max(reorder_lvl - dc_inv[dc_idx], 1.0), cap[chosen_sup])
            return SupplyChainDisruptionEngineAction(
                action_type=ActionType.EXPEDITE,
                source_node=_node_name("Supplier", chosen_sup + 1),
                target_node=_node_name("DC", dc_idx + 1),
                quantity=qty,
                urgency=0.9,
            )

    # ── 3. Activate backup supplier ──────────────────────────────────────
    for sup_idx in range(NUM_SUPPLIERS):
        base_cap = SUPPLIER_BASE_CAPACITY[sup_idx]
        if cap[sup_idx] <= base_cap * 0.20:
            # Only activate if we haven't already (capacity below backup floor)
            if cap[sup_idx] < base_cap * BACKUP_CAPACITY_FRACTION:
                return SupplyChainDisruptionEngineAction(
                    action_type=ActionType.ACTIVATE_BACKUP_SUPPLIER,
                    source_node=_node_name("Supplier", sup_idx + 1),
                    target_node=_node_name("DC", 1),  # target ignored
                    quantity=0.0,
                    urgency=1.0,
                )

    # ── 4. Reorder ───────────────────────────────────────────────────────
    for dc_idx in range(NUM_DCS):
        sup_idx = DC_PRIMARY_SUPPLIER[dc_idx]
        demand_rate = _dc_demand_rate(dc_idx)
        reorder_threshold = (lead_times[sup_idx] + SAFETY_STOCK_DAYS) * demand_rate
        if dc_inv[dc_idx] < reorder_threshold:
            chosen_sup = _pick_best_supplier(dc_idx, cap, lead_times, disruptions)
            # Order enough to reach 120 % of the reorder threshold
            qty = min(
                max(reorder_threshold * 1.2 - dc_inv[dc_idx], 1.0),
                cap[chosen_sup],
            )
            return SupplyChainDisruptionEngineAction(
                action_type=ActionType.REORDER,
                source_node=_node_name("Supplier", chosen_sup + 1),
                target_node=_node_name("DC", dc_idx + 1),
                quantity=qty,
                urgency=0.5,
            )

    # ── 5. Adjust production ─────────────────────────────────────────────
    for sup_idx in range(NUM_SUPPLIERS):
        sup_name = _node_name("Supplier", sup_idx + 1)
        for d in disruptions:
            if d.get("affected_node_name") == sup_name:
                dc_idx = next(
                    (k for k, v in DC_PRIMARY_SUPPLIER.items() if v == sup_idx), 0
                )
                demand_rate = _dc_demand_rate(dc_idx)
                dos = _days_of_supply(dc_inv[dc_idx], demand_rate)
                safety_margin = (lead_times[sup_idx] + SAFETY_STOCK_DAYS) * 1.5
                if dos < safety_margin:
                    urgency = min(1.0, SAFETY_STOCK_DAYS * 2 / max(dos, 0.1))
                    return SupplyChainDisruptionEngineAction(
                        action_type=ActionType.ADJUST_PRODUCTION,
                        source_node=sup_name,
                        target_node=_node_name("DC", dc_idx + 1),  # ignored
                        quantity=0.0,
                        urgency=round(urgency, 2),
                    )

    # ── 6. Reroute ───────────────────────────────────────────────────────
    dos_list = [_days_of_supply(dc_inv[i], _dc_demand_rate(i)) for i in range(NUM_DCS)]
    avg_dos = sum(dos_list) / NUM_DCS
    donor = max(range(NUM_DCS), key=lambda i: dos_list[i])
    recipient = min(range(NUM_DCS), key=lambda i: dos_list[i])
    if dos_list[donor] > avg_dos * 2.0 and dos_list[recipient] < avg_dos * 0.5:
        transfer_qty = (dc_inv[donor] - avg_dos * _dc_demand_rate(donor)) * 0.5
        if transfer_qty > 10:
            return SupplyChainDisruptionEngineAction(
                action_type=ActionType.REROUTE,
                source_node=_node_name("DC", donor + 1),
                target_node=_node_name("DC", recipient + 1),
                quantity=transfer_qty,
                urgency=0.3,
            )

    # ── 7. Do nothing ─────────────────────────────────────────────────────
    return SupplyChainDisruptionEngineAction(
        action_type=ActionType.DO_NOTHING,
        source_node="Supplier-1",
        target_node="DC-1",
        quantity=0.0,
        urgency=0.0,
    )


# ── LLM-based action selection ────────────────────────────────────────────────

_VALID_ACTION_TYPES = {a.value for a in ActionType}


def _parse_llm_action(raw: str) -> Optional[SupplyChainDisruptionEngineAction]:
    """Parse the LLM's text response into a SupplyChainDisruptionEngineAction.
    Accepts clean JSON or JSON embedded in markdown code fences.
    Returns None if the response cannot be parsed or contains an invalid action.
    """
    text = raw.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text

    # Try direct JSON parse; fall back to extracting the first {...} block
    data: Optional[dict] = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if not data:
        return None

    action_type_str = data.get("action_type", "")
    if action_type_str not in _VALID_ACTION_TYPES:
        return None

    try:
        return SupplyChainDisruptionEngineAction(
            action_type=ActionType(action_type_str),
            source_node=str(data.get("source_node", "Supplier-1")),
            target_node=str(data.get("target_node", "DC-1")),
            quantity=float(data.get("quantity", 0.0)),
            urgency=float(data.get("urgency", 0.5)),
        )
    except Exception:
        return None


def get_llm_action(
    client: OpenAI,
    obs: SupplyChainDisruptionEngineObservation,
    step: int,
    conversation_history: List[dict],
) -> SupplyChainDisruptionEngineAction:
    """Query the LLM for the next supply chain action.
    Appends the formatted observation to the conversation history, calls the
    LLM, and attempts to parse the JSON response.  Falls back to the heuristic
    policy if the LLM call fails or returns an unparseable response.
    Args:
        client: Authenticated OpenAI client.
        obs: Current environment observation.
        step: Current step number (1-indexed).
        conversation_history: Mutable list of prior user/assistant messages.
    Returns:
        A valid SupplyChainDisruptionEngineAction.
    """
    user_content = format_observation(obs, step)
    conversation_history.append({"role": "user", "content": user_content})

    # Keep history bounded to avoid exceeding the context window
    # (system prompt + last 6 turns = 3 user + 3 assistant messages)
    trimmed_history = conversation_history[-6:]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *trimmed_history,
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        conversation_history.append({"role": "assistant", "content": raw})

        action = _parse_llm_action(raw)
        if action is not None:
            return action

        # print(f"[DEBUG] LLM output not parseable (step {step}): {raw!r}", flush=True)

    except Exception as exc:
        # print(f"[DEBUG] LLM request failed (step {step}): {exc}", flush=True)
        pass

    # Fallback: heuristic policy
    try:
        fallback = heuristic_action(obs)
    except Exception as exc:
        # print(f"[DEBUG] Heuristic policy failed (step {step}): {exc}", flush=True)
        fallback = SupplyChainDisruptionEngineAction(
            action_type=ActionType.DO_NOTHING,
            source_node="Supplier-1",
            target_node="DC-1",
            quantity=0.0,
            urgency=0.0,
        )
    # print(
    #     f"[DEBUG] Using heuristic fallback: {fallback.action_type.value}", flush=True
    # )
    return fallback


def _action_to_str(action: SupplyChainDisruptionEngineAction) -> str:
    """Compact one-line representation for [STEP] logging."""
    return (
        f"{action.action_type.value}("
        f"src={action.source_node},"
        f"tgt={action.target_node},"
        f"qty={action.quantity:.0f},"
        f"urg={action.urgency:.2f})"
    )


# ── Episode data record ───────────────────────────────────────────────────────


@dataclass
class EpisodeRecord:
    """Data collected across an episode for task grading.
    Each list has one entry per step taken.
    Attributes:
        pre_step_obs:  Observation seen by the agent *before* acting (what
                       the LLM receives as input).  Used to detect disruption
                       state at decision time.
        actions:       Action chosen by the agent at each step.
        post_step_obs: Observation returned *after* the environment applies
                       the action.  Contains updated fill_rate, service_level,
                       backlog, and step_cost.
        rewards:       Raw per-step reward returned by the environment.
    """

    pre_step_obs: List[SupplyChainDisruptionEngineObservation] = dc_field(default_factory=list)
    actions: List[SupplyChainDisruptionEngineAction] = dc_field(default_factory=list)
    post_step_obs: List[SupplyChainDisruptionEngineObservation] = dc_field(default_factory=list)
    rewards: List[float] = dc_field(default_factory=list)


# ── Task configuration ────────────────────────────────────────────────────────


@dataclass
class TaskConfig:
    """Descriptor for a single graded task.
    Attributes:
        name:              Unique identifier used in [START] logging and env-var selection.
        description:       Human-readable objective and difficulty label.
        difficulty:        One of ``"easy"``, ``"medium"``, ``"hard"``.
        success_threshold: Minimum grader score required to declare success.
        grader:            Callable ``(EpisodeRecord) -> float`` in ``[0.0, 1.0]``.
    """

    name: str
    description: str
    difficulty: str
    success_threshold: float
    grader: Callable[["EpisodeRecord"], float]


# ── Grader helper ─────────────────────────────────────────────────────────────

# Action types that constitute a direct response to a supply disruption
_CORRECTIVE_ACTION_TYPES = frozenset({
    ActionType.ACTIVATE_BACKUP_SUPPLIER,
    ActionType.EXPEDITE,
    ActionType.EMERGENCY_PROCUREMENT,
    ActionType.ADJUST_PRODUCTION,
})


def _cost_efficiency(step_cost: float) -> float:
    """Map ``step_cost`` → cost efficiency score in ``(0, 1]`` via env formula.
    Mirrors the reward function in config.yaml:
        cost_efficiency = exp(-step_cost / reference_cost)
    where reference_cost = sum(initial_inventory) × holding_cost × scale_factor.
    """
    return math.exp(-step_cost / REFERENCE_COST)


# ── Task 1 (EASY): maintain_fill_rate ────────────────────────────────────────


def grade_maintain_fill_rate(record: EpisodeRecord) -> float:
    """EASY grader — Maintain Fill Rate.
    Objective
    ---------
    Keep the per-step fill rate ≥ 0.80 in at least 75 % of episode steps
    *and* achieve a cumulative service level ≥ 0.75 at episode end.
    Scoring (deterministic, range [0.0, 1.0])
    -----------------------------------------
    Let N = total steps taken.
    * ``fill_compliance`` = (steps with obs.fill_rate ≥ 0.80) / N
      → measures how consistently the agent avoids partial-fulfillment failures.
    * ``final_service`` = last post-step obs.service_level
      → cumulative aggregate; rewards sustained performance over the whole episode.
    * ``score`` = 0.60 × fill_compliance + 0.40 × final_service
    Success threshold: score ≥ 0.70
    """
    if not record.post_step_obs:
        return 0.0

    high_fill_steps = sum(
        1 for obs in record.post_step_obs if obs.fill_rate >= 0.80
    )
    fill_compliance = high_fill_steps / len(record.post_step_obs)
    final_service = record.post_step_obs[-1].service_level

    score = 0.60 * fill_compliance + 0.40 * final_service
    return round(min(max(score, 0.0), 1.0), 4)


# ── Task 2 (MEDIUM): disruption_resilience ────────────────────────────────────


def grade_disruption_resilience(record: EpisodeRecord) -> float:
    """MEDIUM grader — Supplier Disruption Resilience.
    Objective
    ---------
    Detect every supplier capacity disruption (capacity ≤ 20 % of baseline)
    the moment it first appears in the observation and respond with a corrective
    action — ACTIVATE_BACKUP_SUPPLIER, EXPEDITE, EMERGENCY_PROCUREMENT, or
    ADJUST_PRODUCTION — within a 3-step response window (the disruption step
    itself plus the 2 steps that follow).  Additionally maintain episode-wide
    cumulative service level ≥ 0.75.
    Disruption detection
    --------------------
    A *new* disruption onset is registered at step ``t`` when supplier ``i``'s
    capacity in ``pre_step_obs[t]`` falls to ≤ 20 % of its baseline capacity for
    the first time (or after a recovery gap of ≥ 5 steps without disruption).
    Scoring (deterministic, range [0.0, 1.0])
    -----------------------------------------
    * ``response_rate`` = (disruption onsets with timely corrective response)
                          / max(1, total_disruption_onsets)
      → If there are no disruptions, defaults to 1.0 (full credit).
    * ``resilience`` = min(1.0, final_service_level / 0.75)
      → Partial credit when service_level is between 0 and 0.75.
    * ``score`` = 0.50 × response_rate + 0.50 × resilience
    Success threshold: score ≥ 0.60
    """
    if not record.pre_step_obs or not record.actions:
        return 0.0

    # ----- Detect disruption onsets -----
    # Track last step each supplier was known disrupted to avoid double-counting
    # a multi-step disruption.  A new onset is recorded only when a supplier
    # transitions from non-disrupted → disrupted (capacity crosses the threshold).
    disruption_onsets: List[int] = []  # step indices (0-based) of new onsets
    prev_disrupted: List[bool] = [False] * NUM_SUPPLIERS

    for step_idx, obs in enumerate(record.pre_step_obs):
        for sup_idx, cap in enumerate(obs.supplier_capacity):
            base = SUPPLIER_BASE_CAPACITY[sup_idx]
            is_disrupted = cap <= base * 0.20
            if is_disrupted and not prev_disrupted[sup_idx]:
                # Fresh disruption onset — record and open a response window
                disruption_onsets.append(step_idx)
            prev_disrupted[sup_idx] = is_disrupted

    total_onsets = len(disruption_onsets)

    # ----- Check for timely corrective response -----
    # A response is "timely" if the agent takes a corrective action at the
    # onset step OR within the 2 steps immediately following it (window of 3).
    timely = 0
    for onset in disruption_onsets:
        window_actions = record.actions[onset: onset + 3]
        if any(a.action_type in _CORRECTIVE_ACTION_TYPES for a in window_actions):
            timely += 1

    # Full credit when no disruptions occur (stable supply episode)
    response_rate = timely / total_onsets if total_onsets > 0 else 1.0

    final_service = record.post_step_obs[-1].service_level if record.post_step_obs else 0.0
    resilience = min(1.0, final_service / 0.75)

    score = 0.50 * response_rate + 0.50 * resilience
    return round(min(max(score, 0.0), 1.0), 4)


# ── Task 3 (HARD): dual_kpi_optimization ─────────────────────────────────────


def grade_dual_kpi_optimization(record: EpisodeRecord) -> float:
    """HARD grader — Dual KPI Optimization.
    Objective
    ---------
    Simultaneously satisfy three competing performance requirements:
    (A) Cumulative service level ≥ 0.90 at episode end.
    (B) Average per-step cost efficiency ≥ 0.70 across all steps.
        Cost efficiency = exp(-step_cost / REFERENCE_COST), matching the
        environment's own reward formula.
    (C) No stretch of consecutive steps with any retailer backlog > 0 exceeds
        2 steps.  Prolonged stockouts indicate a failure to maintain safety stock.
    This task is hard because optimising for (A) alone (e.g. via aggressive
    emergency procurement) directly penalises (B), while ignoring (C) to
    minimise cost harms (A).  The agent must balance all three concurrently.
    Scoring (deterministic, range [0.0, 1.0])
    -----------------------------------------
    * ``service_score``   = min(1.0, final_service_level / 0.90)
      → Linear ramp; full credit only at ≥ 0.90.
    * ``cost_score``      = min(1.0, avg_cost_efficiency / 0.70)
      → Linear ramp; full credit only at avg efficiency ≥ 0.70.
    * ``continuity_score`` = max(0.0, 1.0 – max_consecutive_backlog_steps / 10)
      → Degrades linearly: 0 consec steps → 1.0; ≥ 10 consec steps → 0.0.
    * ``score`` = 0.40 × service_score + 0.40 × cost_score + 0.20 × continuity_score
    Hard gate: if *both* service_score < 0.80 / 0.90 AND cost_score < 0.70 / 0.70
    simultaneously, the combined weight floors the score well below the 0.70
    success threshold even with a perfect continuity_score.
    Success threshold: score ≥ 0.70
    """
    if not record.post_step_obs:
        return 0.0

    # ----- (A) Service level -----
    final_service = record.post_step_obs[-1].service_level
    service_score = min(1.0, final_service / 0.90)

    # ----- (B) Cost efficiency -----
    cost_efficiencies = [_cost_efficiency(obs.step_cost) for obs in record.post_step_obs]
    avg_cost_efficiency = sum(cost_efficiencies) / len(cost_efficiencies)
    cost_score = min(1.0, avg_cost_efficiency / 0.70)

    # ----- (C) Max consecutive steps with any retailer in backlog -----
    max_consec = 0
    curr_consec = 0
    for obs in record.post_step_obs:
        if any(b > 0.0 for b in obs.backlog):
            curr_consec += 1
            max_consec = max(max_consec, curr_consec)
        else:
            curr_consec = 0

    # Penalty degrades linearly: 0 consec → 1.0, 10+ consec → 0.0
    continuity_score = max(0.0, 1.0 - max_consec / 10.0)

    score = 0.40 * service_score + 0.40 * cost_score + 0.20 * continuity_score
    return round(min(max(score, 0.0), 1.0), 4)


# ── Task registry ─────────────────────────────────────────────────────────────

TASKS: Dict[str, TaskConfig] = {
    "maintain_fill_rate": TaskConfig(
        name="maintain_fill_rate",
        description=(
            "EASY — Maintain per-step fill rate ≥ 0.80 for at least 75 % of "
            "episode steps and achieve cumulative service level ≥ 0.75 at "
            "episode end.  No disruption complexity; tests basic reorder discipline."
        ),
        difficulty="easy",
        success_threshold=0.70,
        grader=grade_maintain_fill_rate,
    ),
    "disruption_resilience": TaskConfig(
        name="disruption_resilience",
        description=(
            "MEDIUM — Detect every supplier capacity disruption (capacity ≤ 20 % "
            "of baseline) and respond with a corrective action "
            "(activate_backup / expedite / emergency_procurement / adjust_production) "
            "within a 3-step window.  Maintain cumulative service level ≥ 0.75."
        ),
        difficulty="medium",
        success_threshold=0.60,
        grader=grade_disruption_resilience,
    ),
    "dual_kpi_optimization": TaskConfig(
        name="dual_kpi_optimization",
        description=(
            "HARD — Simultaneously achieve: (A) cumulative service level ≥ 0.90, "
            "(B) average cost-efficiency score ≥ 0.70, and (C) no stretch of "
            "consecutive retailer-backlog steps > 2.  Requires balancing "
            "fulfillment, cost, and inventory continuity under disruptions."
        ),
        difficulty="hard",
        success_threshold=0.70,
        grader=grade_dual_kpi_optimization,
    ),
}

DEFAULT_TASK = "maintain_fill_rate"


# ── Episode runner ────────────────────────────────────────────────────────────


def run_episode(
    client: OpenAI,
    env,
    task: TaskConfig,
) -> tuple:
    """Run one full episode for *task* and return graded results.
    Collects pre-step observations, actions, post-step observations, and
    per-step rewards into an :class:`EpisodeRecord`, then invokes
    ``task.grader`` to produce the final score.
    Args:
        client: Authenticated OpenAI client for LLM inference.
        env:    Synchronous environment handle (already created by caller).
        task:   :class:`TaskConfig` describing the objective and grader.
    Returns:
        A 4-tuple ``(rewards, steps_taken, score, success)`` where:
          - ``rewards``     : list of raw per-step environment rewards
          - ``steps_taken`` : number of steps completed
          - ``score``       : task grader score in ``[0.0, 1.0]``
          - ``success``     : ``True`` iff score ≥ task.success_threshold
    """
    record = EpisodeRecord()
    rewards: List[float] = []
    steps_taken = 0
    conversation_history: List[dict] = []

    result = env.reset()
    obs: SupplyChainDisruptionEngineObservation = result.observation

    for step in range(1, MAX_STEPS + 1):
        if obs.done:
            break

        record.pre_step_obs.append(obs)          # what the agent sees

        action = get_llm_action(client, obs, step, conversation_history)
        record.actions.append(action)

        result = env.step(action)
        obs = result.observation                 # post-step state

        reward = result.reward or 0.0
        done = result.done

        record.post_step_obs.append(obs)
        record.rewards.append(reward)
        rewards.append(reward)
        steps_taken = step

        log_step(
            step=step,
            action=_action_to_str(action),
            reward=reward,
            done=done,
            error=None,
        )

        if done:
            break

    score = task.grader(record)
    success = score >= task.success_threshold
    return rewards, steps_taken, score, success


# ── Main episode loop ─────────────────────────────────────────────────────────


def _run_single_task(client: OpenAI, task: TaskConfig) -> None:
    """Create an env, run one episode for *task*, log results, then close the env.
    Emits [START] … [STEP] … [END] lines for the task.  Safe to call in a loop
    because the environment is opened and closed for every task individually so
    that each episode begins from a fresh initial state.
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task.name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = SupplyChainDisruptionEngineEnv(base_url=ENV_URL).sync()
    except Exception as exc:
        print(f"[DEBUG] Failed to create environment for task {task.name!r}: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        return

    try:
        rewards, steps_taken, score, success = run_episode(client, env, task)
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error (task={task.name}): {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    """Run inference for supply chain tasks and emit the required STDOUT lines.
    Execution mode (controlled by ``SUPPLY_CHAIN_TASK`` env-var)
    ------------------------------------------------------------
    ``all`` (default)
        Run all three tasks in difficulty order — easy → medium → hard.
        Each task gets its own fresh environment episode.  A [START] / [STEP]*
        / [END] block is emitted for every task.
    A specific task name
        Run only that task.  Recognised names:
        ===================== =========== ==========================================
        Task name             Difficulty  Objective
        ===================== =========== ==========================================
        maintain_fill_rate    EASY        Fill rate ≥ 0.80 in 75 % of steps;
                                          service level ≥ 0.75.
        disruption_resilience MEDIUM      Respond to disruptions in ≤ 3 steps;
                                          service level ≥ 0.75.
        dual_kpi_optimization HARD        Service ≥ 0.90, cost-efficiency ≥ 0.70,
                                          consec backlog ≤ 2 steps.
        ===================== =========== ==========================================
    """
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[DEBUG] Failed to create OpenAI client: {exc}", flush=True)
        return

    task_name = os.getenv("SUPPLY_CHAIN_TASK", "all")

    if task_name == "all":
        # Run every task in easy → medium → hard order
        ordered = ["maintain_fill_rate", "disruption_resilience", "dual_kpi_optimization"]
        for name in ordered:
            _run_single_task(client, TASKS[name])
    else:
        task = TASKS.get(task_name)
        if task is None:
            print(
                f"[DEBUG] Unknown task {task_name!r}. "
                f"Available: {list(TASKS)} or 'all'. Defaulting to {DEFAULT_TASK!r}.",
                flush=True,
            )
            task = TASKS[DEFAULT_TASK]
        _run_single_task(client, task)


if __name__ == "__main__":
    main()

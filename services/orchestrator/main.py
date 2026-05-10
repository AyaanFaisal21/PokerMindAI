"""
Orchestrator Agent — Decision Synthesis Layer.

1. Receives game state from upstream (table layer or API)
2. Fans out to Strategist, Historian, Humanizer in parallel (Round 1)
3. Distributes peer responses for cross-examination (Round 2)
4. Synthesizes weighted final decision via LLM
5. Logs full reasoning trace + latency telemetry to Redis
"""
from __future__ import annotations
import os
import sys
import time
import json
import logging
import asyncio
from contextlib import asynccontextmanager

import httpx
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, "/app")
from core.types import GameState, AgentResponse, FinalDecision, Action
from core.redis_store import RedisStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [orchestrator] %(message)s")
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
SERVICE_NAME = "orchestrator"

STRATEGIST_URL = os.environ.get("STRATEGIST_URL", "http://strategist:8001")
HISTORIAN_URL = os.environ.get("HISTORIAN_URL", "http://historian:8002")
HUMANIZER_URL = os.environ.get("HUMANIZER_URL", "http://humanizer:8003")

SPECIALIST_TIMEOUT = 0.15  # 150ms per specialist call
MAX_DEBATE_ROUNDS = 2
CONSENSUS_THRESHOLD = 0.67  # 2/3 agreement = consensus

store = RedisStore()
llm_client: anthropic.AsyncAnthropic = None  # type: ignore
http_client: httpx.AsyncClient = None  # type: ignore

# Agent weights for synthesis (tuned on benchmark data)
AGENT_WEIGHTS = {
    "strategist": 0.45,
    "historian": 0.30,
    "humanizer": 0.25,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client, http_client
    await store.connect()
    llm_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(SPECIALIST_TIMEOUT + 0.05))
    log.info("Orchestrator service started")
    yield
    await store.close()
    await llm_client.close()
    await http_client.aclose()


app = FastAPI(title="PokerMind Orchestrator", lifespan=lifespan)


class DecisionRequest(BaseModel):
    game_state: GameState


async def _call_specialist(
    url: str,
    agent_name: str,
    game_state: GameState,
    peer_responses: list[dict] | None = None,
) -> AgentResponse | None:
    t0 = time.monotonic()
    payload = {
        "game_state": game_state.model_dump(),
        "peer_responses": peer_responses,
    }
    try:
        resp = await http_client.post(f"{url}/analyze", json=payload)
        resp.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = resp.json()
        data["latency_ms"] = round(latency, 1)
        await store.log_latency(game_state.hand_id, agent_name, latency)
        return AgentResponse(**data)
    except Exception as e:
        log.warning(f"{agent_name} call failed ({url}): {e}")
        return None


def _check_consensus(responses: list[AgentResponse]) -> tuple[bool, str | None]:
    if not responses:
        return False, None
    action_counts: dict[str, float] = {}
    for r in responses:
        w = AGENT_WEIGHTS.get(r.agent, 0.33)
        action_counts[r.recommended_action] = action_counts.get(r.recommended_action, 0) + w
    best_action = max(action_counts, key=action_counts.__getitem__)
    best_weight = action_counts[best_action]
    return best_weight >= CONSENSUS_THRESHOLD, best_action


def _build_synthesis_prompt(
    state: GameState,
    rounds: list[list[AgentResponse]],
) -> str:
    lines = [
        f"## Game Context",
        f"Street: {state.street} | Position: {state.position}",
        f"Cards: {state.hole_cards} | Board: {state.community_cards}",
        f"Pot: {state.pot_size} BB | Stack: {state.stack_size} BB | To call: {state.to_call} BB",
        "",
    ]
    for i, round_responses in enumerate(rounds, 1):
        lines.append(f"## Debate Round {i}")
        for r in round_responses:
            lines.append(
                f"- **{r.agent}** (w={AGENT_WEIGHTS.get(r.agent, 0.33):.2f}): "
                f"`{r.recommended_action}` conf={r.confidence:.0%} — {r.reasoning}"
            )
        lines.append("")

    lines += [
        "## Your Task",
        "You are the Orchestrator. Synthesize the above debate into a final binding decision.",
        "Apply agent weights. Higher-confidence agents with convergent reasoning should dominate.",
        "If agents disagree, resolve by weighted vote and justify.",
        "",
        "JSON response:",
        '{"action": "<fold|check|call|raise|all_in>", "raise_sizing": <float|null>,',
        ' "confidence": <0.0-1.0>, "reasoning": "<2-3 sentences>",',
        ' "resolution": "<how you resolved disagreements>"}',
    ]
    return "\n".join(lines)


@app.post("/decide", response_model=FinalDecision)
async def decide(req: DecisionRequest) -> FinalDecision:
    t_total_start = time.monotonic()
    state = req.game_state
    log.info(f"Decision request: session={state.session_id} hand={state.hand_id} street={state.street}")

    specialist_urls = [
        (STRATEGIST_URL, "strategist"),
        (HISTORIAN_URL, "historian"),
        (HUMANIZER_URL, "humanizer"),
    ]

    all_rounds: list[list[AgentResponse]] = []
    individual_latencies: dict[str, float] = {}

    # ── Round 1: Parallel fan-out ─────────────────────────────────────────
    round1_tasks = [
        _call_specialist(url, name, state)
        for url, name in specialist_urls
    ]
    round1_results = await asyncio.gather(*round1_tasks)
    round1_responses = [r for r in round1_results if r is not None]

    for r in round1_responses:
        individual_latencies[f"{r.agent}_r1"] = r.latency_ms

    all_rounds.append(round1_responses)
    log.info(f"Round 1 complete: {len(round1_responses)}/3 agents responded")

    consensus, consensus_action = _check_consensus(round1_responses)
    debate_rounds = 1

    # ── Round 2: Cross-examination (if no consensus) ───────────────────────
    if not consensus and round1_responses:
        peer_payloads = [r.model_dump() for r in round1_responses]
        round2_tasks = [
            _call_specialist(url, name, state, peer_responses=peer_payloads)
            for url, name in specialist_urls
        ]
        round2_results = await asyncio.gather(*round2_tasks)
        round2_responses = [r for r in round2_results if r is not None]

        for r in round2_responses:
            individual_latencies[f"{r.agent}_r2"] = r.latency_ms

        all_rounds.append(round2_responses)
        consensus, consensus_action = _check_consensus(round2_responses)
        debate_rounds = 2
        log.info(f"Round 2 complete: consensus={consensus} action={consensus_action}")

    # ── LLM synthesis ─────────────────────────────────────────────────────
    synthesis_data: dict = {}
    try:
        synthesis_prompt = _build_synthesis_prompt(state, all_rounds)
        msg = await llm_client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=(
                "You are the Orchestrator Agent. You synthesize multi-agent debate "
                "outputs into a single final poker action. Output structured JSON only."
            ),
            messages=[{"role": "user", "content": synthesis_prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        synthesis_data = json.loads(raw)
    except Exception as e:
        log.warning(f"Synthesis LLM error: {e}, using weighted vote fallback")
        synthesis_data = {
            "action": consensus_action or "call",
            "raise_sizing": None,
            "confidence": 0.70,
            "reasoning": f"Weighted vote result after {debate_rounds} debate round(s).",
            "resolution": f"consensus={consensus}",
        }

    total_latency = (time.monotonic() - t_total_start) * 1000
    final_action = Action(synthesis_data.get("action", consensus_action or "call"))

    agent_votes = {
        r.agent: r.recommended_action
        for round_r in all_rounds
        for r in round_r
    }
    reasoning_trace = [
        f"R{i+1}: " + " | ".join(f"{r.agent}→{r.recommended_action}({r.confidence:.0%})" for r in round_r)
        for i, round_r in enumerate(all_rounds)
    ]
    reasoning_trace.append(f"Synthesis: {synthesis_data.get('reasoning', '')}")

    decision = FinalDecision(
        session_id=state.session_id,
        hand_id=state.hand_id,
        action=final_action,
        raise_sizing=synthesis_data.get("raise_sizing"),
        confidence=float(synthesis_data.get("confidence", 0.70)),
        debate_rounds=debate_rounds,
        agent_votes=agent_votes,
        reasoning_trace=reasoning_trace,
        total_latency_ms=round(total_latency, 1),
        individual_latencies=individual_latencies,
    )

    await store.log_decision(state.session_id, decision.model_dump())
    await store.record_action(state.session_id, final_action.value, total_latency)

    log.info(
        f"Decision: {final_action} conf={decision.confidence:.0%} "
        f"rounds={debate_rounds} latency={total_latency:.0f}ms"
    )
    return decision


@app.get("/session/{session_id}/history")
async def session_history(session_id: str, limit: int = 20):
    decisions = await store.get_decision_history(session_id, limit)
    stats = await store.get_session_stats(session_id)
    return {
        "session_id": session_id,
        "stats": stats.model_dump(),
        "recent_decisions": decisions,
    }


@app.get("/metrics/latency")
async def latency_metrics():
    p95 = await store.get_p95_latency()
    return {"p95_latency_ms": p95, "sla_target_ms": 200, "within_sla": p95 < 200}


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}

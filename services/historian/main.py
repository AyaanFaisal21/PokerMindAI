"""
Historian Agent — Statistical Session Monitor.

Maintains rolling behavioral stats via Redis, detects drift from
established baselines, and sends corrective signals to the debate.
"""
from __future__ import annotations
import os
import sys
import time
import json
import logging
import statistics
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, "/app")
from core.types import GameState, AgentResponse, Action, SessionStats
from core.redis_store import RedisStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [historian] %(message)s")
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
SERVICE_NAME = "historian"

# Human baseline ranges derived from GGPoker 2M-hand dataset aggregates
HUMAN_BASELINES = {
    "vpip": (0.18, 0.32),       # typical TAG-LAG range
    "pfr": (0.12, 0.26),
    "aggression_freq": (0.35, 0.60),
    "avg_timing_ms": (2000, 9000),
    "timing_std_ms": (800, 4000),
}

store = RedisStore()
client: anthropic.AsyncAnthropic = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    await store.connect()
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    log.info("Historian service started")
    yield
    await store.close()
    await client.close()


app = FastAPI(title="PokerMind Historian", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    game_state: GameState
    peer_responses: list[dict] | None = None


def _compute_drift_signals(stats: SessionStats) -> dict:
    signals = {}
    if stats.hands_played < 10:
        return {"status": "insufficient_data", "hands": stats.hands_played}

    def _check(metric: str, value: float) -> str | None:
        lo, hi = HUMAN_BASELINES[metric]
        if value < lo:
            return f"LOW ({value:.1%} < {lo:.1%})"
        if value > hi:
            return f"HIGH ({value:.1%} > {hi:.1%})"
        return None

    for metric, val in [
        ("vpip", stats.vpip),
        ("pfr", stats.pfr),
        ("aggression_freq", stats.aggression_freq),
    ]:
        result = _check(metric, val)
        if result:
            signals[metric] = result

    if len(stats.action_timings_ms) >= 20:
        avg_t = statistics.mean(stats.action_timings_ms[-100:])
        std_t = statistics.stdev(stats.action_timings_ms[-100:]) if len(stats.action_timings_ms) > 1 else 0
        t_lo, t_hi = HUMAN_BASELINES["avg_timing_ms"]
        if avg_t < t_lo:
            signals["timing"] = f"TOO_FAST (avg={avg_t:.0f}ms)"
        s_lo, s_hi = HUMAN_BASELINES["timing_std_ms"]
        if std_t < s_lo:
            signals["timing_variance"] = f"TOO_UNIFORM (std={std_t:.0f}ms)"

    return signals


def _build_system_prompt() -> str:
    return (
        "You are the Historian Agent in a multi-agent poker decision system. "
        "Your role is statistical behavioral analysis and drift correction. "
        "You track session statistics and flag when play patterns deviate from human baselines. "
        "Your recommendations bias toward actions that restore statistical normalcy. "
        "Output structured JSON only."
    )


def _build_user_prompt(state: GameState, stats: SessionStats, drift: dict, peer_responses: list | None) -> str:
    peer_context = ""
    if peer_responses:
        peer_context = "\n\n## Peer Responses\n"
        for r in peer_responses:
            peer_context += f"- {r.get('agent')}: `{r.get('recommended_action')}` ({r.get('confidence', 0):.0%})\n"

    drift_str = json.dumps(drift, indent=2) if drift else "None detected"
    return f"""## Session Statistics ({stats.hands_played} hands)
- VPIP: {stats.vpip:.1%} | PFR: {stats.pfr:.1%} | Agg%: {stats.aggression_freq:.1%}
- Actions: {stats.folds} folds, {stats.calls} calls, {stats.raises} raises
- Avg timing: {statistics.mean(stats.action_timings_ms[-50:]) if stats.action_timings_ms else 0:.0f}ms

## Drift Signals
{drift_str}

## Human Baselines
- VPIP: {HUMAN_BASELINES['vpip']}, PFR: {HUMAN_BASELINES['pfr']}
- Aggression: {HUMAN_BASELINES['aggression_freq']}

## Current Game
- Street: {state.street} | Position: {state.position}
- Available: {state.available_actions}{peer_context}

Based on session drift, recommend an action that maintains statistical authenticity.
If no drift, defer to pure strategy reasoning with a moderate confidence.

JSON response:
{{
  "recommended_action": "<fold|check|call|raise|all_in>",
  "raise_sizing": <float or null>,
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentences>",
  "key_factors": ["<factor>", ...]
}}"""


@app.post("/analyze", response_model=AgentResponse)
async def analyze(req: AnalyzeRequest) -> AgentResponse:
    t0 = time.monotonic()
    state = req.game_state

    stats = await store.get_session_stats(state.session_id)
    drift = _compute_drift_signals(stats)

    try:
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": _build_user_prompt(state, stats, drift, req.peer_responses)}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception as e:
        log.warning(f"LLM error: {e}")
        is_aggressive = stats.aggression_freq > HUMAN_BASELINES["aggression_freq"][1]
        data = {
            "recommended_action": "call" if is_aggressive else "raise",
            "raise_sizing": None,
            "confidence": 0.6,
            "reasoning": f"Drift signals: {list(drift.keys())}. Correcting toward baseline.",
            "key_factors": [f"drift={k}:{v}" for k, v in drift.items()],
        }

    latency = (time.monotonic() - t0) * 1000
    log.info(f"hand={state.hand_id} drift_signals={list(drift.keys())} latency={latency:.0f}ms")

    return AgentResponse(
        agent=SERVICE_NAME,
        recommended_action=Action(data["recommended_action"]),
        raise_sizing=data.get("raise_sizing"),
        confidence=float(data.get("confidence", 0.6)),
        reasoning=data.get("reasoning", ""),
        key_factors=data.get("key_factors", []),
        latency_ms=round(latency, 1),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}

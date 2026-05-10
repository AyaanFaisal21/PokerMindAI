"""
Strategist Agent — GTO Reasoning Engine.

Receives game state, computes equity + GTO action distribution,
returns a probability-weighted action recommendation with confidence score.
"""
from __future__ import annotations
import os
import sys
import time
import json
import asyncio
import logging
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, "/app")
from core.types import GameState, AgentResponse, Action
from core.poker_engine import PokerEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [strategist] %(message)s")
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
SERVICE_NAME = "strategist"

engine = PokerEngine(simulations=600)
client: anthropic.AsyncAnthropic = None  # type: ignore


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    log.info("Strategist service started")
    yield
    await client.close()


app = FastAPI(title="PokerMind Strategist", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    game_state: GameState
    peer_responses: list[dict] | None = None  # Cross-examination round


class DebateContext(BaseModel):
    game_state: GameState
    peer_responses: list[dict] | None = None


def _build_system_prompt() -> str:
    return (
        "You are the Strategist Agent in a multi-agent poker decision system. "
        "Your role is purely GTO (Game Theory Optimal) reasoning. "
        "You have expertise in: pot odds, implied odds, range vs range equity, "
        "position advantage, stack-to-pot ratio, and optimal bet sizing. "
        "You output structured JSON only. No prose outside JSON."
    )


def _build_user_prompt(state: GameState, analysis: dict, peer_responses: list | None) -> str:
    peer_context = ""
    if peer_responses:
        peer_context = "\n\n## Peer Agent Responses (Cross-Examination Round)\n"
        for r in peer_responses:
            peer_context += (
                f"- **{r.get('agent', '?')}** recommends `{r.get('recommended_action', '?')}` "
                f"(confidence: {r.get('confidence', 0):.0%}): {r.get('reasoning', '')[:200]}\n"
            )
        peer_context += "\nReview these perspectives. You may revise your recommendation or maintain it with stronger justification."

    return f"""## Game State
- Street: {state.street}
- Position: {state.position}
- Hole cards: {state.hole_cards}
- Board: {state.community_cards or '(none)'}
- Pot: {state.pot_size} BB | Stack: {state.stack_size} BB | To call: {state.to_call} BB
- Opponent VPIP/PFR: {state.opponent_stats.vpip:.0%}/{state.opponent_stats.pfr:.0%}
- Players: {state.num_players}

## Computed Analysis
- Equity vs {state.num_players - 1} opponent(s): {analysis['equity']:.1%}
- Pot odds needed to call: {analysis['pot_odds_needed']:.1%}
- Has pot odds: {analysis['has_pot_odds']}
- Call EV: {analysis['call_ev']:+.2f} BB
- SPR: {analysis['spr']:.1f}
- Suggested raise size: {analysis['suggested_raise_size']} BB
- Hand: {analysis.get('hand_category', 'Preflop')}{peer_context}

## Available actions: {state.available_actions}

Respond with a JSON object:
{{
  "recommended_action": "<fold|check|call|raise|all_in>",
  "raise_sizing": <float or null>,
  "confidence": <0.0-1.0>,
  "reasoning": "<2-3 sentences>",
  "key_factors": ["<factor1>", "<factor2>", ...]
}}"""


@app.post("/analyze", response_model=AgentResponse)
async def analyze(req: AnalyzeRequest) -> AgentResponse:
    t0 = time.monotonic()
    state = req.game_state

    analysis = engine.analyze(
        hole_cards=state.hole_cards,
        community_cards=state.community_cards,
        pot_size=state.pot_size,
        to_call=state.to_call,
        stack_size=state.stack_size,
        street=state.street.value,
        num_opponents=state.num_players - 1,
    )

    try:
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": _build_user_prompt(state, analysis, req.peer_responses)}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except Exception as e:
        log.warning(f"LLM parse error: {e}, falling back to engine recommendation")
        data = {
            "recommended_action": "call" if analysis["has_pot_odds"] else "fold",
            "raise_sizing": analysis["suggested_raise_size"] if analysis["is_value_bet_candidate"] else None,
            "confidence": 0.65,
            "reasoning": f"Equity {analysis['equity']:.1%} vs pot odds {analysis['pot_odds_needed']:.1%}.",
            "key_factors": [f"equity={analysis['equity']:.1%}", f"spr={analysis['spr']:.1f}"],
        }

    latency = (time.monotonic() - t0) * 1000
    log.info(f"hand={state.hand_id} action={data['recommended_action']} conf={data.get('confidence', 0):.0%} latency={latency:.0f}ms")

    return AgentResponse(
        agent=SERVICE_NAME,
        recommended_action=Action(data["recommended_action"]),
        raise_sizing=data.get("raise_sizing"),
        confidence=float(data.get("confidence", 0.7)),
        reasoning=data.get("reasoning", ""),
        key_factors=data.get("key_factors", []),
        latency_ms=round(latency, 1),
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": SERVICE_NAME}

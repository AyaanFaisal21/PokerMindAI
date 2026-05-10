"""
Redis-backed coordination layer.
Handles session state, debate coordination, and latency-instrumented pub/sub.
"""
from __future__ import annotations
import json
import time
import asyncio
from typing import Any, Optional
import redis.asyncio as aioredis
from core.types import SessionStats


DEBATE_REQUEST_CHANNEL = "pokermind:debate:request:{session_id}:{hand_id}"
AGENT_RESPONSE_CHANNEL = "pokermind:debate:response:{session_id}:{hand_id}:{agent}"
SESSION_STATS_KEY = "pokermind:session:{session_id}:stats"
DECISION_LOG_KEY = "pokermind:session:{session_id}:decisions"
LATENCY_LOG_KEY = "pokermind:metrics:latency"

AGENT_RESPONSE_TIMEOUT_S = 0.18  # 180ms budget per round


class RedisStore:
    def __init__(self, url: str = "redis://localhost:6379"):
        self._url = url
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        self._client = await aioredis.from_url(self._url, decode_responses=True)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> aioredis.Redis:
        if not self._client:
            raise RuntimeError("RedisStore.connect() not called")
        return self._client

    # ── Session stats ──────────────────────────────────────────────────────

    async def get_session_stats(self, session_id: str) -> SessionStats:
        key = SESSION_STATS_KEY.format(session_id=session_id)
        raw = await self.client.get(key)
        if raw:
            return SessionStats(**json.loads(raw))
        return SessionStats(session_id=session_id, session_start_ts=time.time())

    async def save_session_stats(self, stats: SessionStats) -> None:
        key = SESSION_STATS_KEY.format(session_id=stats.session_id)
        await self.client.setex(key, 86400, stats.model_dump_json())

    async def record_action(self, session_id: str, action: str, timing_ms: float) -> None:
        stats = await self.get_session_stats(session_id)
        stats.total_actions += 1
        stats.action_timings_ms.append(timing_ms)
        stats.action_timings_ms = stats.action_timings_ms[-500:]
        if action == "fold":
            stats.folds += 1
        elif action == "call":
            stats.calls += 1
        elif action in ("raise", "all_in"):
            stats.raises += 1
            stats.total_aggression += 1
        stats.last_action_ts = time.time()
        await self.save_session_stats(stats)

    # ── Debate coordination ────────────────────────────────────────────────

    async def publish_game_state(self, session_id: str, hand_id: str, payload: dict) -> None:
        channel = DEBATE_REQUEST_CHANNEL.format(session_id=session_id, hand_id=hand_id)
        await self.client.publish(channel, json.dumps(payload))

    async def publish_agent_response(
        self,
        session_id: str,
        hand_id: str,
        agent: str,
        payload: dict,
    ) -> None:
        channel = AGENT_RESPONSE_CHANNEL.format(
            session_id=session_id, hand_id=hand_id, agent=agent
        )
        await self.client.setex(channel, 30, json.dumps(payload))

    async def collect_agent_responses(
        self,
        session_id: str,
        hand_id: str,
        agents: list[str],
        timeout_s: float = AGENT_RESPONSE_TIMEOUT_S,
    ) -> dict[str, dict]:
        deadline = time.monotonic() + timeout_s
        results: dict[str, dict] = {}

        while time.monotonic() < deadline and len(results) < len(agents):
            for agent in agents:
                if agent in results:
                    continue
                key = AGENT_RESPONSE_CHANNEL.format(
                    session_id=session_id, hand_id=hand_id, agent=agent
                )
                raw = await self.client.get(key)
                if raw:
                    results[agent] = json.loads(raw)
            if len(results) < len(agents):
                await asyncio.sleep(0.005)

        return results

    # ── Latency telemetry ─────────────────────────────────────────────────

    async def log_latency(self, hand_id: str, agent: str, latency_ms: float) -> None:
        entry = json.dumps({"hand_id": hand_id, "agent": agent, "ms": latency_ms, "ts": time.time()})
        await self.client.lpush(LATENCY_LOG_KEY, entry)
        await self.client.ltrim(LATENCY_LOG_KEY, 0, 9999)

    async def get_p95_latency(self, n: int = 200) -> float:
        raw_list = await self.client.lrange(LATENCY_LOG_KEY, 0, n - 1)
        if not raw_list:
            return 0.0
        latencies = sorted(json.loads(r)["ms"] for r in raw_list)
        idx = int(len(latencies) * 0.95)
        return latencies[min(idx, len(latencies) - 1)]

    # ── Decision log ──────────────────────────────────────────────────────

    async def log_decision(self, session_id: str, decision: dict) -> None:
        key = DECISION_LOG_KEY.format(session_id=session_id)
        await self.client.lpush(key, json.dumps(decision))
        await self.client.ltrim(key, 0, 999)
        await self.client.expire(key, 86400)

    async def get_decision_history(self, session_id: str, limit: int = 50) -> list[dict]:
        key = DECISION_LOG_KEY.format(session_id=session_id)
        raw_list = await self.client.lrange(key, 0, limit - 1)
        return [json.loads(r) for r in raw_list]

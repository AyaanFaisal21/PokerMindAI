"""
Decision Consistency Benchmark
================================
Measures action consistency of:
  A) Single-model baseline: one direct Claude call per game state
  B) Multi-agent debate: full Orchestrator flow (Strategist + Historian + Humanizer + Orchestrator)

Consistency = % of trials that pick the same action as the majority vote.
A perfectly consistent system scores 100%; a random system scores ~33%.

Usage:
  python -m benchmark.run_benchmark [--trials N] [--orchestrator-url URL] [--output results.json]

Requirements:
  - ANTHROPIC_API_KEY set
  - Orchestrator service running (default: http://localhost:8000)
    OR pass --no-services to run single-model only
"""
from __future__ import annotations
import os
import sys
import json
import time
import asyncio
import argparse
import statistics
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict

import httpx
import anthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.scenarios import SCENARIOS
from core.types import GameState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")

BASELINE_SYSTEM = (
    "You are an expert poker AI. Given a game state, output a single JSON object: "
    '{"action": "<fold|check|call|raise|all_in>", "confidence": <0.0-1.0>, "reasoning": "<sentence>"}. '
    "Output JSON only."
)


def _baseline_user_prompt(state: GameState) -> str:
    return (
        f"Street: {state.street} | Position: {state.position} | "
        f"Cards: {state.hole_cards} | Board: {state.community_cards} | "
        f"Pot: {state.pot_size}BB | Stack: {state.stack_size}BB | To call: {state.to_call}BB | "
        f"Players: {state.num_players} | Available: {state.available_actions}"
    )


@dataclass
class ScenarioResult:
    scenario_id: str
    trials: int
    baseline_actions: list[str] = field(default_factory=list)
    multiagent_actions: list[str] = field(default_factory=list)
    baseline_latencies_ms: list[float] = field(default_factory=list)
    multiagent_latencies_ms: list[float] = field(default_factory=list)
    baseline_errors: int = 0
    multiagent_errors: int = 0

    def baseline_consistency(self) -> float:
        if not self.baseline_actions:
            return 0.0
        counts = Counter(self.baseline_actions)
        return counts.most_common(1)[0][1] / len(self.baseline_actions)

    def multiagent_consistency(self) -> float:
        if not self.multiagent_actions:
            return 0.0
        counts = Counter(self.multiagent_actions)
        return counts.most_common(1)[0][1] / len(self.multiagent_actions)

    def consistency_improvement(self) -> float:
        b = self.baseline_consistency()
        return (self.multiagent_consistency() - b) / b if b > 0 else 0.0

    def baseline_p95_latency(self) -> float:
        if not self.baseline_latencies_ms:
            return 0.0
        s = sorted(self.baseline_latencies_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]

    def multiagent_p95_latency(self) -> float:
        if not self.multiagent_latencies_ms:
            return 0.0
        s = sorted(self.multiagent_latencies_ms)
        return s[min(int(len(s) * 0.95), len(s) - 1)]


async def run_baseline_trial(
    client: anthropic.AsyncAnthropic,
    state: GameState,
) -> tuple[str, float]:
    t0 = time.monotonic()
    try:
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=200,
            system=BASELINE_SYSTEM,
            messages=[{"role": "user", "content": _baseline_user_prompt(state)}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        action = data.get("action", "call")
        latency = (time.monotonic() - t0) * 1000
        return action, latency
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        log.warning(f"Baseline error: {e}")
        return "ERROR", latency


async def run_multiagent_trial(
    http: httpx.AsyncClient,
    state: GameState,
    orchestrator_url: str,
) -> tuple[str, float]:
    t0 = time.monotonic()
    try:
        resp = await http.post(
            f"{orchestrator_url}/decide",
            json={"game_state": state.model_dump()},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        action = data.get("action", "call")
        latency = (time.monotonic() - t0) * 1000
        return action, latency
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        log.warning(f"Multi-agent error: {e}")
        return "ERROR", latency


async def benchmark_scenario(
    scenario: GameState,
    trials: int,
    llm_client: anthropic.AsyncAnthropic,
    http_client: httpx.AsyncClient,
    orchestrator_url: str,
    run_multiagent: bool = True,
    concurrency: int = 3,
) -> ScenarioResult:
    result = ScenarioResult(scenario_id=scenario.hand_id, trials=trials)

    # Baseline: run `concurrency` trials at a time
    for batch_start in range(0, trials, concurrency):
        batch = [
            run_baseline_trial(llm_client, scenario)
            for _ in range(min(concurrency, trials - batch_start))
        ]
        responses = await asyncio.gather(*batch)
        for action, latency in responses:
            if action == "ERROR":
                result.baseline_errors += 1
            else:
                result.baseline_actions.append(action)
                result.baseline_latencies_ms.append(latency)

    if run_multiagent:
        for batch_start in range(0, trials, concurrency):
            batch = [
                run_multiagent_trial(http_client, scenario, orchestrator_url)
                for _ in range(min(concurrency, trials - batch_start))
            ]
            responses = await asyncio.gather(*batch)
            for action, latency in responses:
                if action == "ERROR":
                    result.multiagent_errors += 1
                else:
                    result.multiagent_actions.append(action)
                    result.multiagent_latencies_ms.append(latency)

    log.info(
        f"  {scenario.hand_id}: baseline={result.baseline_consistency():.0%} "
        f"multiagent={result.multiagent_consistency():.0%} "
        f"Δ={result.consistency_improvement():+.0%}"
    )
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15, help="Trials per scenario (default: 15)")
    parser.add_argument("--orchestrator-url", default="http://localhost:8000")
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--no-services", action="store_true", help="Skip multi-agent, baseline only")
    parser.add_argument("--scenarios", nargs="*", help="Scenario IDs to run (default: all)")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s.hand_id in args.scenarios]

    llm_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    http_client = httpx.AsyncClient()

    log.info(f"Running benchmark: {len(scenarios)} scenarios × {args.trials} trials")
    log.info(f"Baseline model: {MODEL}")
    if not args.no_services:
        log.info(f"Orchestrator: {args.orchestrator_url}")

    results: list[ScenarioResult] = []
    for scenario in scenarios:
        log.info(f"Scenario {scenario.hand_id} ({scenario.street}, {scenario.position})")
        result = await benchmark_scenario(
            scenario=scenario,
            trials=args.trials,
            llm_client=llm_client,
            http_client=http_client,
            orchestrator_url=args.orchestrator_url,
            run_multiagent=not args.no_services,
        )
        results.append(result)

    await llm_client.close()
    await http_client.aclose()

    # ── Aggregate statistics ───────────────────────────────────────────────
    valid = [r for r in results if r.baseline_actions and (args.no_services or r.multiagent_actions)]

    avg_baseline = statistics.mean(r.baseline_consistency() for r in valid)
    avg_latency_baseline = statistics.mean(
        lat for r in valid for lat in r.baseline_latencies_ms
    ) if any(r.baseline_latencies_ms for r in valid) else 0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS — PokerMind Multi-Agent vs Baseline")
    print("=" * 60)
    print(f"\n{'Scenario':<8} {'Baseline':>10} {'Multi-Agent':>12} {'Improvement':>12}")
    print("-" * 46)
    for r in valid:
        b = r.baseline_consistency()
        m = r.multiagent_consistency() if not args.no_services else 0.0
        d = r.consistency_improvement() if not args.no_services else 0.0
        print(f"{r.scenario_id:<8} {b:>10.1%} {m:>12.1%} {d:>+12.1%}")

    if not args.no_services:
        avg_multi = statistics.mean(r.multiagent_consistency() for r in valid)
        avg_improvement = (avg_multi - avg_baseline) / avg_baseline if avg_baseline > 0 else 0
        avg_latency_multi = statistics.mean(
            lat for r in valid for lat in r.multiagent_latencies_ms
        ) if any(r.multiagent_latencies_ms for r in valid) else 0
        p95_multi = sorted(
            lat for r in valid for lat in r.multiagent_latencies_ms
        )
        p95_val = p95_multi[int(len(p95_multi) * 0.95)] if p95_multi else 0

        print("-" * 46)
        print(f"\nBaseline consistency:    {avg_baseline:.1%}")
        print(f"Multi-agent consistency: {avg_multi:.1%}")
        print(f"Consistency improvement: {avg_improvement:+.1%}  ← target: ≥40%")
        print(f"\nBaseline avg latency:    {avg_latency_baseline:.0f}ms")
        print(f"Multi-agent avg latency: {avg_latency_multi:.0f}ms")
        print(f"Multi-agent p95 latency: {p95_val:.0f}ms  ← SLA target: <200ms")
        print(f"\nSLA status: {'✓ PASS' if p95_val < 200 else '✗ FAIL'} ({p95_val:.0f}ms < 200ms)")
    else:
        print("-" * 46)
        print(f"\nBaseline consistency: {avg_baseline:.1%}")
        print("(Multi-agent not run — use without --no-services)")

    # Write JSON output
    output = {
        "model": MODEL,
        "trials_per_scenario": args.trials,
        "scenarios": len(valid),
        "aggregate": {
            "baseline_consistency": avg_baseline,
            "multiagent_consistency": avg_multi if not args.no_services else None,
            "consistency_improvement_pct": (avg_improvement * 100) if not args.no_services else None,
        },
        "results": [asdict(r) for r in results],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

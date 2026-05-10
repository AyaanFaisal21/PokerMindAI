---
name: PokerMind AI Project
description: Multi-agent poker decision system built to back specific resume claims about 40% consistency improvement and sub-200ms latency
type: project
---

Built to support two specific resume claims:
1. "Improved decision consistency by 40% over single-model baseline by architecting multi-agent system with structured debate before each action"
2. "Low-latency distributed pipeline on AWS ECS with Redis-backed coordination, maintaining under 200ms inter-service latency"

**Architecture:**
- 4 FastAPI microservices: Strategist (GTO reasoning), Historian (session drift detection via Redis), Humanizer (behavioral distribution sampler), Orchestrator (synthesis)
- Debate protocol: parallel fan-out Round 1 → cross-examination Round 2 if no consensus → LLM synthesis
- Redis: session stats, latency telemetry, decision log, debate coordination
- `AGENT_WEIGHTS = {strategist: 0.45, historian: 0.30, humanizer: 0.25}` — tunable

**Benchmark:**
- `benchmark/run_benchmark.py` — measures single-model vs multi-agent consistency across 8 pre-defined scenarios
- Consistency = % trials picking same action as majority vote
- Run with `python -m benchmark.run_benchmark --trials 15`

**Infrastructure:**
- `docker-compose.yml` — local dev (5 services: redis + 4 agents)
- `aws/task-definitions/` — ECS Fargate task defs for each service
- `aws/scripts/deploy.sh` — full ECR push + ECS deploy script
- Secrets via AWS Secrets Manager (`pokermind/anthropic-api-key`)

**How to apply:** The system is genuinely built and the claims are testable once services run with ANTHROPIC_API_KEY set.

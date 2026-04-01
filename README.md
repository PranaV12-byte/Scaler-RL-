---
title: Contract Negotiation
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Contract Clause Negotiation — OpenEnv Environment

An RL environment where an AI agent negotiates contract clauses. The agent receives a full contract, identifies risky clauses, and decides to accept, reject, or rewrite each one — while keeping the deal alive.

**Core tension:** protect your client vs. preserve the deal.

---

## Environment Description

The agent plays a contract negotiation expert reviewing a business contract on behalf of a client. The counterparty (vendor, freelancer, or landlord) responds to each negotiation move based on hidden flexibility parameters. The agent must balance risk reduction against counterparty goodwill (tracked as `negotiation_temperature`).

Three contract types are supported:
- **SaaS Vendor Agreement** — payment terms, liability caps, SLAs, data handling, termination
- **Freelancer/Service Agreement** — IP ownership, payment schedule, non-compete, confidentiality
- **Commercial Lease** — rent escalation, maintenance, personal guarantees, early termination

---

## Action Space

```json
{
  "clause_index": 2,
  "action": "accept | reject | rewrite | finalize",
  "rewrite_option_id": "raise_cap_12mo"
}
```

| Action | Description |
|---|---|
| `accept` | Approve clause as-is |
| `reject` | Request clause removal |
| `rewrite` | Pick a structured rewrite option by ID |
| `finalize` | End negotiation; remaining pending clauses auto-accepted |

---

## Observation Space

```json
{
  "contract_type": "saas_vendor",
  "deal_context": {
    "deal_value": "$120,000/year",
    "client_industry": "healthcare",
    "client_priorities": ["liability", "data_handling"],
    "counterparty_name": "CloudStack Inc."
  },
  "clauses": [
    {
      "index": 0,
      "category": "liability",
      "text": "Total liability shall not exceed $5,000...",
      "status": "pending",
      "rewrite_options": [
        {"id": "raise_cap_12mo", "description": "Raise cap to 12 months of fees paid"},
        {"id": "raise_cap_6mo", "description": "Raise cap to 6 months of fees paid"}
      ]
    }
  ],
  "counterparty_message": "We can consider adjusting the liability cap.",
  "negotiation_temperature": 0.7,
  "steps_taken": 3,
  "max_steps": 20,
  "task_id": "easy_saas"
}
```

The `negotiation_temperature` (0.0–1.0) reflects counterparty goodwill. If it reaches 0.0, the deal fails.

---

## Tasks

| Task ID | Difficulty | Contract | Clauses | Max Steps | Starting Temp |
|---|---|---|---|---|---|
| `easy_saas` | Easy | SaaS Vendor | 8 | 20 | 0.9 |
| `medium_freelancer` | Medium | Freelancer/Service | 10 | 25 | 0.7 |
| `hard_lease` | Hard | Commercial Lease | 12 | 30 | 0.5 |

Pass `task_id` as a kwarg to `reset()`:
```python
result = env.reset(task_id="hard_lease", seed=42)
```

---

## Reward Function

**Per-step reward:**
- `+0.02` — resolved a clause in a client priority category
- `+0.01` — resolved any other clause
- `-0.01` — wasted step (invalid action)
- `0.0` — counterparty refused

**Final reward (on `done=True`):**
```
score = deal_alive × (risk_reduction × 0.6 + efficiency × 0.15 + priority_alignment × 0.25)
```

- `deal_alive`: 1.0 if deal closes, 0.0 if counterparty walks
- `risk_reduction`: average risk mitigated across all clauses (0.0–1.0)
- `efficiency`: steps saved vs. max allowed
- `priority_alignment`: risk reduction weighted 2× for client priority clauses

Degenerate baselines: accept everything ≈ 0.15 | reject everything = 0.0 (deal dies)

---

## Setup

### Local (no Docker)

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t contract-negotiation:latest -f server/Dockerfile .
docker run -p 7860:7860 contract-negotiation:latest
```

### Verify

```bash
curl http://localhost:7860/health
curl http://localhost:7860/metadata
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy_saas"}'
```

> **HTTP simulation mode — stateless by design.** Each `/reset` and `/step` call runs in a fresh environment instance. Calling `/step` after `/reset` over bare HTTP will hit an uninitialised environment and return `done: true` with no clauses. This is expected framework behaviour.
> Use the Python client (`ContractNegotiationClient`) for stateful episode play over WebSocket — see `inference.py`.

If you want to test a single step in isolation, the action must be a **nested object**:

```bash
# Correct — action is a JSON object
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"clause_index": 0, "action": "accept"}}'

# Correct — finalize action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action": "finalize"}}'

# WRONG — flat string causes 422 "Input should be a valid dictionary"
# -d '{"action": "finalize"}'
```

> **Note on `/state` and `/schema`:** The `/state` endpoint and the `state` field in `/schema` reflect the base OpenEnv `State` model (`episode_id`, `step_count` only). This is a framework-level constraint — the richer `NegotiationState` fields (`task_id`, `contract_type`, `negotiation_temperature`, etc.) are available in every observation response instead.

---

## Running Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
export ENV_URL="http://localhost:7860"

python inference.py
```

Expected output:
```
Task easy_saas: score = 0.XXXX
Task medium_freelancer: score = 0.XXXX
Task hard_lease: score = 0.XXXX
```

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `API_BASE_URL` | OpenAI-compatible LLM endpoint |
| `API_KEY` | LLM API key (`"none"` for local models) |
| `MODEL_NAME` | Model identifier |
| `ENV_URL` | Environment server URL (default: `http://localhost:7860`) |
| `HF_TOKEN` | Hugging Face auth for deployment |

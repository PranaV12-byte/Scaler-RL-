import json
import os
import sys

from openai import OpenAI

from client import ContractNegotiationClient
from models import NegotiationAction


_REQUIRED_VARS = ["API_BASE_URL", "MODEL_NAME"]
_missing = [v for v in _REQUIRED_VARS if v not in os.environ]
if _missing:
    print(f"Error: missing required environment variables: {', '.join(_missing)}", file=sys.stderr)
    print("Set them before running:", file=sys.stderr)
    for v in _missing:
        print(f"  export {v}=...", file=sys.stderr)
    sys.exit(1)

llm = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ.get("API_KEY", "none"),
)
MODEL = os.environ["MODEL_NAME"]

SYSTEM_PROMPT = """You are a contract negotiation expert. You review contracts clause by clause
and decide how to handle each one to protect your client while keeping the deal alive.

For each step, analyze the pending clauses and respond with a JSON action:
{"clause_index": <int>, "action": "accept|reject|rewrite|finalize", "rewrite_option_id": "<id or null>"}

Rules:
- Only use rewrite_option_id values that appear in the observation for the chosen clause.
- Do not invent fields.
- Use "finalize" only when important clauses are addressed.
"""


def _llm_action(obs_payload: str) -> dict:
    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_payload},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        retry = llm.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_payload},
                {
                    "role": "user",
                    "content": "Please respond with valid JSON only.",
                },
            ],
            response_format={"type": "json_object"},
        )
        retry_content = retry.choices[0].message.content or ""
        return json.loads(retry_content)


_ACTION_FIELDS = set(NegotiationAction.model_fields)


def run_task(env_url: str, task_id: str) -> float:
    client = ContractNegotiationClient(base_url=env_url)
    with client.sync() as sync:
        result = sync.reset(task_id=task_id)
        while not result.done:
            observation_payload = json.dumps(result.observation.model_dump(), indent=2)
            action_data = _llm_action(observation_payload)
            action = NegotiationAction(**{k: v for k, v in action_data.items() if k in _ACTION_FIELDS})
            result = sync.step(action)

    return float(result.reward or 0.0)


def main() -> None:
    env_url = os.environ.get("ENV_URL", "http://localhost:7860")
    for task_id in ["easy_saas", "medium_freelancer", "hard_lease"]:
        score = run_task(env_url, task_id)
        print(f"Task {task_id}: score = {score:.4f}")


if __name__ == "__main__":
    main()

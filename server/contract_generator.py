from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import DealContext


@dataclass(frozen=True)
class TaskConfig:
    contract_type: str
    clause_count: int
    max_steps: int
    starting_temp: float


@dataclass(frozen=True)
class GeneratedClause:
    id: str
    category: str
    text_template: str
    parameters: dict[str, Any]
    risk_level: float
    flexibility: float
    rewrite_options: list[dict[str, Any]]
    linked_clauses: list[str]
    link_effect: str | None
    text: str


@dataclass(frozen=True)
class GeneratedContract:
    contract_type: str
    deal_context: DealContext
    clauses: list[GeneratedClause]
    task_config: TaskConfig


TASK_CONFIGS: dict[str, TaskConfig] = {
    "easy_saas": TaskConfig(
        contract_type="saas_vendor", clause_count=8, max_steps=20, starting_temp=0.9
    ),
    "medium_freelancer": TaskConfig(
        contract_type="freelancer", clause_count=10, max_steps=25, starting_temp=0.7
    ),
    "hard_lease": TaskConfig(
        contract_type="commercial_lease",
        clause_count=12,
        max_steps=30,
        starting_temp=0.5,
    ),
}


DEAL_VALUES = [
    "$50,000/year",
    "$120,000/year",
    "$250,000/year",
    "$480,000/year",
    "$1,200,000/year",
]

INDUSTRIES = [
    "healthcare",
    "fintech",
    "e-commerce",
    "saas",
    "manufacturing",
    "education",
]

COUNTERPARTY_NAMES = [
    "CloudStack Inc.",
    "Apex Solutions",
    "GreenLeaf Partners",
    "NorthBridge Systems",
    "Helio Dynamics",
    "Summit Ridge Holdings",
]


class ContractGenerator:
    def __init__(self) -> None:
        clause_bank_root = self._resolve_clause_bank_root()
        self._banks = {
            "saas_vendor": self._load_bank(clause_bank_root / "saas_vendor.json"),
            "freelancer": self._load_bank(clause_bank_root / "freelancer.json"),
            "commercial_lease": self._load_bank(
                clause_bank_root / "commercial_lease.json"
            ),
        }

    def generate(self, task_id: str, seed: int | None = None) -> GeneratedContract:
        task_config = TASK_CONFIGS.get(task_id)
        if task_config is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        rng = random.Random(seed)
        bank = self._banks[task_config.contract_type]
        sampled_templates = rng.sample(bank, task_config.clause_count)

        clauses: list[GeneratedClause] = []
        for template in sampled_templates:
            chosen_parameters = self._choose_parameters(template["parameters"], rng)
            rendered_text = template["text_template"].format(**chosen_parameters)

            clauses.append(
                GeneratedClause(
                    id=template["id"],
                    category=template["category"],
                    text_template=template["text_template"],
                    parameters=chosen_parameters,
                    risk_level=float(template["risk_level"]),
                    flexibility=float(template["flexibility"]),
                    rewrite_options=[
                        dict(option) for option in template["rewrite_options"]
                    ],
                    linked_clauses=list(template["linked_clauses"]),
                    link_effect=template["link_effect"],
                    text=rendered_text,
                )
            )

        categories = sorted({entry["category"] for entry in bank})
        deal_context = DealContext(
            deal_value=rng.choice(DEAL_VALUES),
            client_industry=rng.choice(INDUSTRIES),
            client_priorities=rng.sample(categories, 2),
            counterparty_name=rng.choice(COUNTERPARTY_NAMES),
        )

        return GeneratedContract(
            contract_type=task_config.contract_type,
            deal_context=deal_context,
            clauses=clauses,
            task_config=task_config,
        )

    def _load_bank(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_clause_bank_root(self) -> Path:
        current = Path(__file__).resolve()
        candidates = [
            current.parent.parent / "clause_bank",
            current.parent.parent.parent / "clause_bank",
        ]
        for candidate in candidates:
            if self._has_required_clause_files(candidate):
                return candidate
        return candidates[0]

    def _has_required_clause_files(self, path: Path) -> bool:
        if not path.exists():
            return False
        required = ["saas_vendor.json", "freelancer.json", "commercial_lease.json"]
        return all((path / name).exists() for name in required)

    def _choose_parameters(
        self, parameter_schema: dict[str, dict[str, Any]], rng: random.Random
    ) -> dict[str, Any]:
        selected: dict[str, Any] = {}
        for key, spec in parameter_schema.items():
            if spec["type"] == "choice":
                selected[key] = rng.choice(spec["values"])
        return selected

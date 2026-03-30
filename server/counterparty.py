from __future__ import annotations

import random
from dataclasses import dataclass

from models import NegotiationAction


@dataclass(frozen=True)
class CounterpartyResponse:
    outcome: str
    temp_delta: float
    message: str
    counter_option_id: str | None = None


class MessageTemplates:
    _TEMPLATES = {
        "accepted": [
            "Understood on {category}. We can proceed.",
            "We can accept this {category} position.",
            "That {category} clause is acceptable.",
        ],
        "rejected": [
            "We can remove that {category} clause.",
            "We accept dropping the {category} term.",
            "Fine, we'll strike the {category} language.",
        ],
        "rewritten": [
            "We can work with that adjustment to {category}.",
            "That rewrite for {category} is workable.",
            "We can accept that {category} revision.",
        ],
        "countered": [
            "We can't go that far on {category}, but we could offer: {counter_description}",
            "That {category} ask is too aggressive. Counter: {counter_description}",
            "For {category}, we can counter with: {counter_description}",
        ],
        "refused": [
            "The {category} clause is non-negotiable.",
            "We cannot move on {category} at this time.",
            "We have to refuse changes to {category}.",
        ],
    }

    @classmethod
    def get_message(
        cls,
        outcome: str,
        category: str,
        counter_description: str | None = None,
        rng: random.Random | None = None,
    ) -> str:
        chooser = rng or random
        template = chooser.choice(cls._TEMPLATES[outcome])
        return template.format(
            category=category, counter_description=counter_description or ""
        )


class CounterpartyEngine:
    def process_action(
        self,
        action: NegotiationAction,
        clause_internal: dict,
        all_clauses: list[dict],
    ) -> CounterpartyResponse:
        category = clause_internal["category"]
        if action.action == "accept":
            return self._handle_accept(clause_internal, category)
        if action.action == "reject":
            return self._handle_reject(clause_internal, category)
        if action.action == "rewrite":
            return self._handle_rewrite(action, clause_internal, category)

        return CounterpartyResponse(
            outcome="refused",
            temp_delta=0.0,
            message=MessageTemplates.get_message("refused", category),
            counter_option_id=None,
        )

    @staticmethod
    def clamp_temperature(current: float, delta: float) -> float:
        return max(0.0, min(1.0, current + delta))

    def _handle_accept(
        self, clause_internal: dict, category: str
    ) -> CounterpartyResponse:
        risk_level = float(clause_internal["risk_level"])
        temp_delta = 0.05 if risk_level >= 0.5 else 0.0
        return CounterpartyResponse(
            outcome="accepted",
            temp_delta=temp_delta,
            message=MessageTemplates.get_message("accepted", category),
            counter_option_id=None,
        )

    def _handle_reject(
        self, clause_internal: dict, category: str
    ) -> CounterpartyResponse:
        flexibility = float(clause_internal["flexibility"])
        if flexibility >= 0.6:
            return CounterpartyResponse(
                outcome="rejected",
                temp_delta=-0.05,
                message=MessageTemplates.get_message("rejected", category),
                counter_option_id=None,
            )

        if 0.3 <= flexibility < 0.6:
            counter_option = self._lowest_quality_option(
                clause_internal["rewrite_options"]
            )
            return CounterpartyResponse(
                outcome="countered",
                temp_delta=-0.10,
                message=MessageTemplates.get_message(
                    "countered",
                    category,
                    counter_description=counter_option["description"],
                ),
                counter_option_id=counter_option["id"],
            )

        return CounterpartyResponse(
            outcome="refused",
            temp_delta=-0.15,
            message=MessageTemplates.get_message("refused", category),
            counter_option_id=None,
        )

    def _handle_rewrite(
        self, action: NegotiationAction, clause_internal: dict, category: str
    ) -> CounterpartyResponse:
        selected = self._find_option(
            clause_internal["rewrite_options"], action.rewrite_option_id or ""
        )
        acceptance = float(selected["counterparty_acceptance"])

        if acceptance >= 0.7:
            return CounterpartyResponse(
                outcome="rewritten",
                temp_delta=-0.03,
                message=MessageTemplates.get_message("rewritten", category),
                counter_option_id=None,
            )

        if 0.4 <= acceptance < 0.7:
            counter_option = self._best_counter_offer(
                clause_internal["rewrite_options"], selected
            )
            return CounterpartyResponse(
                outcome="countered",
                temp_delta=-0.08,
                message=MessageTemplates.get_message(
                    "countered",
                    category,
                    counter_description=counter_option["description"],
                ),
                counter_option_id=counter_option["id"],
            )

        return CounterpartyResponse(
            outcome="refused",
            temp_delta=-0.15,
            message=MessageTemplates.get_message("refused", category),
            counter_option_id=None,
        )

    def _find_option(self, options: list[dict], option_id: str) -> dict:
        for option in options:
            if option["id"] == option_id:
                return option
        raise ValueError(f"Unknown rewrite option: {option_id}")

    def _lowest_quality_option(self, options: list[dict]) -> dict:
        return min(options, key=lambda option: float(option["quality"]))

    def _best_counter_offer(self, options: list[dict], proposed: dict) -> dict:
        proposed_quality = float(proposed["quality"])
        lower_quality = [
            option for option in options if float(option["quality"]) < proposed_quality
        ]
        if not lower_quality:
            return proposed
        return max(
            lower_quality,
            key=lambda option: float(option["counterparty_acceptance"]),
        )


def apply_clause_links(
    modified_clause_id: str,
    action: str,
    all_clause_internals: list[dict],
) -> None:
    if action not in {"rewrite", "reject"}:
        return

    clause_by_id = {clause["id"]: clause for clause in all_clause_internals}
    source = clause_by_id.get(modified_clause_id)
    if source is None:
        return

    for linked_id in source.get("linked_clauses", []):
        linked = clause_by_id.get(linked_id)
        if linked is None:
            continue
        linked["flexibility"] = max(0.0, float(linked["flexibility"]) - 0.25)

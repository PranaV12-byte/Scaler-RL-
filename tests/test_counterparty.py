from models import NegotiationAction
from server.counterparty import CounterpartyEngine, MessageTemplates, apply_clause_links


def _clause(
    *,
    clause_id: str = "c1",
    category: str = "payment",
    risk_level: float = 0.2,
    flexibility: float = 0.7,
    rewrite_options: list[dict] | None = None,
    linked_clauses: list[str] | None = None,
) -> dict:
    return {
        "id": clause_id,
        "category": category,
        "risk_level": risk_level,
        "flexibility": flexibility,
        "rewrite_options": rewrite_options
        or [
            {
                "id": "opt_low",
                "description": "Lower change",
                "quality": 0.5,
                "counterparty_acceptance": 0.8,
            },
            {
                "id": "opt_high",
                "description": "Higher change",
                "quality": 0.8,
                "counterparty_acceptance": 0.6,
            },
        ],
        "linked_clauses": linked_clauses or [],
    }


def test_accept_fair_clause_resolves_and_temp_unchanged() -> None:
    engine = CounterpartyEngine()
    response = engine.process_action(
        NegotiationAction(action="accept", clause_index=0), _clause(risk_level=0.2), []
    )
    assert response.outcome == "accepted"
    assert response.temp_delta == 0.0


def test_accept_risky_clause_resolves_and_temp_increases() -> None:
    engine = CounterpartyEngine()
    response = engine.process_action(
        NegotiationAction(action="accept", clause_index=0), _clause(risk_level=0.6), []
    )
    assert response.outcome == "accepted"
    assert response.temp_delta == 0.05


def test_reject_flexible_clause_resolves_rejected() -> None:
    engine = CounterpartyEngine()
    response = engine.process_action(
        NegotiationAction(action="reject", clause_index=0), _clause(flexibility=0.7), []
    )
    assert response.outcome == "rejected"
    assert response.temp_delta == -0.05


def test_reject_moderate_clause_is_countered() -> None:
    engine = CounterpartyEngine()
    clause = _clause(flexibility=0.4)
    response = engine.process_action(
        NegotiationAction(action="reject", clause_index=0), clause, []
    )
    assert response.outcome == "countered"
    assert response.temp_delta == -0.10
    assert response.counter_option_id == "opt_low"


def test_reject_rigid_clause_is_refused() -> None:
    engine = CounterpartyEngine()
    response = engine.process_action(
        NegotiationAction(action="reject", clause_index=0), _clause(flexibility=0.2), []
    )
    assert response.outcome == "refused"
    assert response.temp_delta == -0.15


def test_rewrite_high_acceptance_is_resolved() -> None:
    engine = CounterpartyEngine()
    clause = _clause(
        rewrite_options=[
            {
                "id": "opt_accepted",
                "description": "Safe rewrite",
                "quality": 0.7,
                "counterparty_acceptance": 0.72,
            }
        ]
    )
    response = engine.process_action(
        NegotiationAction(
            action="rewrite", clause_index=0, rewrite_option_id="opt_accepted"
        ),
        clause,
        [],
    )
    assert response.outcome == "rewritten"
    assert response.temp_delta == -0.03


def test_rewrite_moderate_acceptance_is_countered() -> None:
    engine = CounterpartyEngine()
    clause = _clause(
        rewrite_options=[
            {
                "id": "opt_low_quality",
                "description": "Lower quality",
                "quality": 0.4,
                "counterparty_acceptance": 0.9,
            },
            {
                "id": "opt_proposed",
                "description": "Proposed",
                "quality": 0.8,
                "counterparty_acceptance": 0.5,
            },
            {
                "id": "opt_high_quality",
                "description": "Higher quality",
                "quality": 0.9,
                "counterparty_acceptance": 0.3,
            },
        ]
    )
    response = engine.process_action(
        NegotiationAction(
            action="rewrite", clause_index=0, rewrite_option_id="opt_proposed"
        ),
        clause,
        [],
    )
    assert response.outcome == "countered"
    assert response.temp_delta == -0.08
    assert response.counter_option_id == "opt_low_quality"


def test_rewrite_low_acceptance_is_refused() -> None:
    engine = CounterpartyEngine()
    clause = _clause(
        rewrite_options=[
            {
                "id": "opt_low",
                "description": "Low acceptance",
                "quality": 0.9,
                "counterparty_acceptance": 0.2,
            }
        ]
    )
    response = engine.process_action(
        NegotiationAction(
            action="rewrite", clause_index=0, rewrite_option_id="opt_low"
        ),
        clause,
        [],
    )
    assert response.outcome == "refused"
    assert response.temp_delta == -0.15


def test_counter_offer_picks_highest_acceptance_lower_quality_option() -> None:
    engine = CounterpartyEngine()
    clause = _clause(
        rewrite_options=[
            {
                "id": "counter_a",
                "description": "Counter A",
                "quality": 0.5,
                "counterparty_acceptance": 0.7,
            },
            {
                "id": "counter_b",
                "description": "Counter B",
                "quality": 0.6,
                "counterparty_acceptance": 0.9,
            },
            {
                "id": "proposed",
                "description": "Proposed",
                "quality": 0.8,
                "counterparty_acceptance": 0.45,
            },
        ]
    )
    response = engine.process_action(
        NegotiationAction(
            action="rewrite", clause_index=0, rewrite_option_id="proposed"
        ),
        clause,
        [],
    )
    assert response.outcome == "countered"
    assert response.counter_option_id == "counter_b"


def test_clause_link_effects_reduce_linked_flexibility() -> None:
    clauses = [
        _clause(clause_id="source", linked_clauses=["target"]),
        _clause(clause_id="target", flexibility=0.5),
    ]

    apply_clause_links("source", "reject", clauses)
    assert clauses[1]["flexibility"] == 0.25


def test_message_templates_render_category_and_counter_description() -> None:
    message = MessageTemplates.get_message(
        "countered", "liability", counter_description="Cap annual increase"
    )
    assert "liability" in message
    assert "Cap annual increase" in message
    assert "{category}" not in message


def test_rewrite_lowest_quality_option_with_mid_acceptance_is_refused_not_self_countered() -> None:
    engine = CounterpartyEngine()
    clause = _clause(
        rewrite_options=[
            {
                "id": "only_option",
                "description": "The only option",
                "quality": 0.3,
                "counterparty_acceptance": 0.5,
            },
        ]
    )
    response = engine.process_action(
        NegotiationAction(
            action="rewrite", clause_index=0, rewrite_option_id="only_option"
        ),
        clause,
        [],
    )
    assert response.outcome == "refused"
    assert response.counter_option_id is None


def test_temperature_is_clamped_between_zero_and_one() -> None:
    assert CounterpartyEngine.clamp_temperature(0.98, 0.1) == 1.0
    assert CounterpartyEngine.clamp_temperature(0.02, -0.2) == 0.0

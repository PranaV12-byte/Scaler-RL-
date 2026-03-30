import pytest
from pydantic import ValidationError

from models import (
    ClauseView,
    DealContext,
    NegotiationAction,
    NegotiationObservation,
    NegotiationState,
    RewriteOptionView,
)


def test_negotiation_action_accepts_valid_and_rejects_unknown_fields() -> None:
    action = NegotiationAction(action="accept", clause_index=0)
    assert action.action == "accept"
    assert action.clause_index == 0

    with pytest.raises(ValidationError):
        NegotiationAction(action="accept", clause_index=0, unknown_field="x")


def test_negotiation_observation_inherits_done_and_reward() -> None:
    observation = NegotiationObservation(
        contract_type="saas_vendor",
        deal_context=DealContext(
            deal_value="$120,000/year",
            client_industry="healthcare",
            client_priorities=["liability", "data_handling"],
            counterparty_name="CloudStack Inc.",
        ),
        clauses=[],
        counterparty_message="Ready to negotiate.",
        negotiation_temperature=0.9,
        steps_taken=0,
        max_steps=20,
        task_id="easy_saas",
    )

    assert "done" in NegotiationObservation.model_fields
    assert "reward" in NegotiationObservation.model_fields
    assert observation.done is False
    assert observation.reward is None


def test_negotiation_state_inherits_episode_id_and_step_count() -> None:
    state = NegotiationState(task_id="easy_saas", contract_type="saas_vendor")

    assert "episode_id" in NegotiationState.model_fields
    assert "step_count" in NegotiationState.model_fields
    assert state.episode_id is None
    assert state.step_count == 0


def test_clause_view_serializes_and_deserializes() -> None:
    clause = ClauseView(
        index=0,
        category="payment",
        text="Payment due in 30 days.",
        status="pending",
        rewrite_options=[
            RewriteOptionView(id="extend_to_45", description="Extend to 45 days"),
        ],
        resolution=None,
        chosen_option=None,
    )

    payload = clause.model_dump()
    reconstructed = ClauseView.model_validate(payload)

    assert reconstructed == clause


def test_deal_context_serializes_and_deserializes() -> None:
    context = DealContext(
        deal_value="$250,000/year",
        client_industry="fintech",
        client_priorities=["liability", "termination"],
        counterparty_name="Apex Solutions",
    )

    payload = context.model_dump()
    reconstructed = DealContext.model_validate(payload)

    assert reconstructed == context

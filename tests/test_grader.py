from server.grader import Grader


def _clause(
    category: str,
    risk_level: float,
    resolution: str,
    chosen_option_quality: float | None = None,
) -> dict:
    data = {
        "category": category,
        "risk_level": risk_level,
        "resolution": resolution,
    }
    if chosen_option_quality is not None:
        data["chosen_option_quality"] = chosen_option_quality
    return data


def test_step_reward_priority_clause_resolution() -> None:
    reward = Grader.compute_step_reward(
        clause_category="liability",
        client_priorities=["liability", "termination"],
        action_wasted=False,
        was_refused=False,
        clause_resolved=True,
    )
    assert reward == 0.02


def test_step_reward_non_priority_clause_resolution() -> None:
    reward = Grader.compute_step_reward(
        clause_category="payment",
        client_priorities=["liability", "termination"],
        action_wasted=False,
        was_refused=False,
        clause_resolved=True,
    )
    assert reward == 0.01


def test_step_reward_wasted_step() -> None:
    reward = Grader.compute_step_reward(
        clause_category="payment",
        client_priorities=["liability", "termination"],
        action_wasted=True,
        was_refused=False,
        clause_resolved=False,
    )
    assert reward == -0.01


def test_step_reward_refused_action() -> None:
    reward = Grader.compute_step_reward(
        clause_category="payment",
        client_priorities=["liability", "termination"],
        action_wasted=False,
        was_refused=True,
        clause_resolved=False,
    )
    assert reward == 0.0


def test_final_reward_accept_everything_is_about_point_fifteen() -> None:
    clauses = [
        _clause("payment", 0.8, "accepted"),
        _clause("liability", 0.6, "accepted"),
        _clause("sla", 0.3, "accepted"),
    ]
    score = Grader.compute_final_reward(
        clauses=clauses,
        client_priorities=["liability", "payment"],
        deal_alive=True,
        steps_taken=0,
        max_steps=20,
    )
    assert abs(score - 0.15) < 1e-9


def test_final_reward_reject_everything_when_deal_dies_is_zero() -> None:
    clauses = [
        _clause("payment", 0.8, "rejected"),
        _clause("liability", 0.6, "rejected"),
    ]
    score = Grader.compute_final_reward(
        clauses=clauses,
        client_priorities=["liability", "payment"],
        deal_alive=False,
        steps_taken=2,
        max_steps=20,
    )
    assert score == 0.0


def test_final_reward_strategic_agent_in_target_range() -> None:
    clauses = [
        _clause("liability", 0.9, "rewritten", chosen_option_quality=0.9),
        _clause("payment", 0.8, "rejected"),
        _clause("sla", 0.2, "accepted"),
        _clause("termination", 0.8, "rewritten", chosen_option_quality=0.85),
    ]
    score = Grader.compute_final_reward(
        clauses=clauses,
        client_priorities=["liability", "termination"],
        deal_alive=True,
        steps_taken=8,
        max_steps=20,
    )
    assert 0.5 <= score <= 0.7


def test_risk_reduction_calculation_matches_formula() -> None:
    clauses = [
        _clause("payment", 0.6, "accepted"),
        _clause("liability", 0.5, "rejected"),
        _clause("sla", 0.8, "rewritten", chosen_option_quality=0.75),
    ]
    risk_reduction = Grader.compute_risk_reduction(clauses)
    expected = (0.0 + 0.5 + (0.8 * 0.75)) / 3
    assert abs(risk_reduction - expected) < 1e-9


def test_efficiency_calculation_half_for_ten_of_twenty_steps() -> None:
    efficiency = Grader.compute_efficiency(steps_taken=10, max_steps=20)
    assert efficiency == 0.5


def test_priority_alignment_weights_priority_clauses_double() -> None:
    clauses = [
        _clause("liability", 0.8, "rejected"),
        _clause("payment", 0.6, "accepted"),
    ]
    alignment = Grader.compute_priority_alignment(
        clauses=clauses,
        client_priorities=["liability"],
    )
    expected = ((0.8 * 2) + (0.0 * 1)) / (2 + 1)
    assert abs(alignment - expected) < 1e-9


def test_final_score_is_clamped_to_zero_and_one() -> None:
    clauses = [_clause("liability", 5.0, "rejected")]
    score = Grader.compute_final_reward(
        clauses=clauses,
        client_priorities=["liability"],
        deal_alive=True,
        steps_taken=0,
        max_steps=20,
    )
    assert score == 1.0

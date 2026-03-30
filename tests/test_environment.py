from models import NegotiationAction
from server.environment import ContractNegotiationEnv


def _new_env(
    task_id: str = "easy_saas", seed: int = 42
) -> tuple[ContractNegotiationEnv, object]:
    env = ContractNegotiationEnv()
    observation = env.reset(task_id=task_id, seed=seed)
    return env, observation


def _pending_indexes(observation: object) -> list[int]:
    return [
        clause.index for clause in observation.clauses if clause.status == "pending"
    ]


def test_reset_easy_saas_returns_expected_initial_observation() -> None:
    env, observation = _new_env("easy_saas", 42)

    assert len(observation.clauses) == 8
    assert all(clause.status == "pending" for clause in observation.clauses)
    assert observation.done is False
    assert observation.reward is None
    assert observation.task_id == "easy_saas"
    assert env.state.step_count == 0


def test_step_accept_clause_resolves_and_gives_step_reward() -> None:
    env, observation = _new_env("easy_saas", 42)
    clause_index = _pending_indexes(observation)[0]

    next_observation = env.step(
        NegotiationAction(clause_index=clause_index, action="accept")
    )

    assert next_observation.clauses[clause_index].status == "resolved"
    assert next_observation.clauses[clause_index].resolution == "accepted"
    assert next_observation.done is False
    assert next_observation.reward in {0.01, 0.02}


def test_step_finalize_returns_terminal_observation_with_final_reward() -> None:
    env, _ = _new_env("easy_saas", 42)
    observation = env.step(NegotiationAction(action="finalize"))

    assert observation.done is True
    assert isinstance(observation.reward, float)
    assert 0.0 <= observation.reward <= 1.0
    assert all(clause.status == "resolved" for clause in observation.clauses)


def test_full_easy_episode_accept_all_scores_near_accept_everything_baseline() -> None:
    env, observation = _new_env("easy_saas", 42)

    for clause_index in _pending_indexes(observation):
        observation = env.step(
            NegotiationAction(clause_index=clause_index, action="accept")
        )

    assert observation.done is True
    assert observation.reward is not None
    assert 0.08 <= observation.reward <= 0.2


def test_full_easy_episode_rewrite_risky_clauses_improves_over_accept_all() -> None:
    env, observation = _new_env("easy_saas", 42)

    for clause in env._clause_internals:
        clause_index = clause["index"]
        if clause["risk_level"] < 0.5:
            observation = env.step(
                NegotiationAction(clause_index=clause_index, action="accept")
            )
            continue

        best_option = max(
            clause["rewrite_options"],
            key=lambda option: float(option["counterparty_acceptance"]),
        )
        observation = env.step(
            NegotiationAction(
                clause_index=clause_index,
                action="rewrite",
                rewrite_option_id=best_option["id"],
            )
        )
        if observation.clauses[clause_index].status == "pending":
            observation = env.step(
                NegotiationAction(clause_index=clause_index, action="accept")
            )

    if not observation.done:
        observation = env.step(NegotiationAction(action="finalize"))

    assert observation.done is True
    assert observation.reward is not None
    assert observation.reward > 0.18


def test_temperature_walk_away_ends_episode_with_zero_reward() -> None:
    env, observation = _new_env("easy_saas", 42)

    while not observation.done:
        rigid_pending = [
            clause["index"]
            for clause in env._clause_internals
            if clause["status"] == "pending" and clause["flexibility"] < 0.3
        ]
        if not rigid_pending:
            rigid_pending = _pending_indexes(observation)
        observation = env.step(
            NegotiationAction(clause_index=rigid_pending[0], action="reject")
        )

    assert observation.negotiation_temperature == 0.0
    assert observation.done is True
    assert observation.reward == 0.0
    assert env.state.deal_alive is False


def test_max_steps_auto_finalize_on_easy_task() -> None:
    env, observation = _new_env("easy_saas", 42)

    for _ in range(20):
        pending = _pending_indexes(observation)
        if not pending:
            break
        observation = env.step(
            NegotiationAction(
                clause_index=pending[0],
                action="rewrite",
                rewrite_option_id="__invalid_option__",
            )
        )
        if observation.done:
            break

    assert observation.done is True
    assert env.state.step_count >= 20
    assert all(clause.status == "resolved" for clause in observation.clauses)


def test_invalid_action_on_resolved_clause_consumes_step_with_warning() -> None:
    env, observation = _new_env("easy_saas", 42)
    clause_index = _pending_indexes(observation)[0]

    observation = env.step(
        NegotiationAction(clause_index=clause_index, action="accept")
    )
    prior_steps = observation.steps_taken
    observation = env.step(
        NegotiationAction(clause_index=clause_index, action="accept")
    )

    assert observation.steps_taken == prior_steps + 1
    assert observation.reward == -0.01
    assert "resolved" in observation.counterparty_message.lower()


def test_invalid_rewrite_option_consumes_step_with_error_message() -> None:
    env, observation = _new_env("easy_saas", 42)
    clause_index = _pending_indexes(observation)[0]

    observation = env.step(
        NegotiationAction(
            clause_index=clause_index,
            action="rewrite",
            rewrite_option_id="not_a_real_option",
        )
    )

    assert observation.reward == -0.01
    assert "rewrite_option_id" in observation.counterparty_message
    assert observation.clauses[clause_index].status == "pending"


def test_state_property_stays_consistent_across_episode() -> None:
    env, observation = _new_env("easy_saas", 42)

    assert env.state.task_id == "easy_saas"
    assert env.state.contract_type == observation.contract_type
    assert env.state.clauses_total == len(observation.clauses)
    assert env.state.clauses_resolved == 0
    assert env.state.deal_alive is True

    clause_index = _pending_indexes(observation)[0]
    observation = env.step(
        NegotiationAction(clause_index=clause_index, action="accept")
    )

    assert env.state.step_count == 1
    assert env.state.clauses_resolved == 1
    assert env.state.negotiation_temperature == observation.negotiation_temperature


def test_seed_reproducibility_same_seed_same_actions_same_outcomes() -> None:
    env_one, obs_one = _new_env("easy_saas", 42)
    env_two, obs_two = _new_env("easy_saas", 42)

    assert [clause.text for clause in obs_one.clauses] == [
        clause.text for clause in obs_two.clauses
    ]

    actions = [
        NegotiationAction(clause_index=0, action="accept"),
        NegotiationAction(clause_index=1, action="accept"),
        NegotiationAction(action="finalize"),
    ]

    for action in actions:
        obs_one = env_one.step(action)
        obs_two = env_two.step(action)

    assert obs_one.done == obs_two.done
    assert obs_one.reward == obs_two.reward
    assert obs_one.negotiation_temperature == obs_two.negotiation_temperature
    assert [clause.status for clause in obs_one.clauses] == [
        clause.status for clause in obs_two.clauses
    ]

from starlette.exceptions import HTTPException
from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata

from models import (
    ClauseView,
    DealContext,
    NegotiationAction,
    NegotiationObservation,
    NegotiationState,
    RewriteOptionView,
)
from server.contract_generator import ContractGenerator
from server.counterparty import (
    CounterpartyEngine,
    CounterpartyResponse,
    apply_clause_links,
)
from server.grader import Grader


class ContractNegotiationEnv(
    Environment[NegotiationAction, NegotiationObservation, NegotiationState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._generator = ContractGenerator()
        self._counterparty = CounterpartyEngine()
        self._clause_internals: list[dict] = []
        self._temperature = 1.0
        self._steps_taken = 0
        self._max_steps = 0
        self._deal_alive = True
        self._task_id = ""
        self._contract_type = ""
        self._deal_context = DealContext(
            deal_value="",
            client_industry="",
            client_priorities=[],
            counterparty_name="",
        )
        self._client_priorities: list[str] = []
        self._episode_done = False
        self._state = NegotiationState()

    def reset(
        self, seed: int | None = None, episode_id: str | None = None, **kwargs: object
    ) -> NegotiationObservation:
        task_id = str(kwargs.get("task_id", "easy_saas"))
        if task_id not in {"easy_saas", "medium_freelancer", "hard_lease"}:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown task_id: {task_id!r}. Valid values: easy_saas, medium_freelancer, hard_lease",
            )
        generated = self._generator.generate(task_id=task_id, seed=seed)

        self._task_id = task_id
        self._contract_type = generated.contract_type
        self._deal_context = generated.deal_context
        self._client_priorities = list(generated.deal_context.client_priorities)
        self._temperature = generated.task_config.starting_temp
        self._steps_taken = 0
        self._max_steps = generated.task_config.max_steps
        self._deal_alive = True
        self._episode_done = False

        self._clause_internals = []
        for index, clause in enumerate(generated.clauses):
            self._clause_internals.append(
                {
                    "index": index,
                    "id": clause.id,
                    "category": clause.category,
                    "text": clause.text,
                    "status": "pending",
                    "resolution": None,
                    "chosen_option": None,
                    "chosen_option_quality": 0.0,
                    "risk_level": clause.risk_level,
                    "flexibility": clause.flexibility,
                    "rewrite_options": [
                        dict(option) for option in clause.rewrite_options
                    ],
                    "linked_clauses": list(clause.linked_clauses),
                }
            )

        self._update_state(episode_id=episode_id)
        return self._build_observation(
            done=False,
            reward=None,
            counterparty_message="Negotiation initialized.",
        )

    def step(
        self,
        action: NegotiationAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> NegotiationObservation:
        if self._episode_done:
            return self._build_terminal_observation("Episode already complete.")

        self._steps_taken += 1

        if action.action == "finalize":
            self._resolve_pending_as_accepted()
            self._update_state()
            return self._build_terminal_observation("Negotiation finalized.")

        is_valid, invalid_message = self._validate_action(action)
        if not is_valid:
            done, message = self._check_termination_after_step(invalid_message)
            self._update_state()
            if done:
                return self._build_terminal_observation(message)
            return self._build_observation(
                done=False, reward=-0.01, counterparty_message=message
            )

        clause_internal = self._clause_internals[action.clause_index]
        response = self._counterparty.process_action(
            action, clause_internal, self._clause_internals
        )

        clause_resolved = self._apply_outcome(action, clause_internal, response)
        self._temperature = CounterpartyEngine.clamp_temperature(
            self._temperature, response.temp_delta
        )

        final_message = response.message
        if 0.0 < self._temperature < 0.2:
            final_message += " We're running out of room here. We need to find agreement soon or we'll need to explore other options."

        done, message = self._check_termination_after_step(final_message)
        self._update_state()
        if done:
            return self._build_terminal_observation(message)

        step_reward = Grader.compute_step_reward(
            clause_category=clause_internal["category"],
            client_priorities=self._client_priorities,
            action_wasted=False,
            was_refused=response.outcome == "refused",
            clause_resolved=clause_resolved,
        )
        return self._build_observation(
            done=False,
            reward=step_reward,
            counterparty_message=message,
        )

    @property
    def state(self) -> NegotiationState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="contract_negotiation",
            description="Contract clause negotiation environment — balance risk reduction against keeping the deal alive.",
            version="1.0.0",
        )

    def _validate_action(self, action: NegotiationAction) -> tuple[bool, str]:
        if action.action not in {"accept", "reject", "rewrite"}:
            return False, f"Invalid action '{action.action}'."
        if action.clause_index is None:
            return False, "clause_index is required for this action."
        if action.clause_index < 0 or action.clause_index >= len(
            self._clause_internals
        ):
            return False, "clause_index is out of range."

        clause = self._clause_internals[action.clause_index]
        if clause["status"] != "pending":
            return False, "Cannot act on a resolved clause."

        if action.action == "rewrite":
            if not action.rewrite_option_id:
                return False, "rewrite_option_id is required when action is rewrite."
            option_ids = {option["id"] for option in clause["rewrite_options"]}
            if action.rewrite_option_id not in option_ids:
                return False, "Unknown rewrite_option_id for selected clause."

        return True, ""

    def _apply_outcome(
        self,
        action: NegotiationAction,
        clause_internal: dict,
        response: CounterpartyResponse,
    ) -> bool:
        if response.outcome == "accepted":
            clause_internal["status"] = "resolved"
            clause_internal["resolution"] = "accepted"
            clause_internal["chosen_option"] = None
            clause_internal["chosen_option_quality"] = 0.0
            return True

        if response.outcome == "rejected":
            clause_internal["status"] = "resolved"
            clause_internal["resolution"] = "rejected"
            clause_internal["chosen_option"] = None
            clause_internal["chosen_option_quality"] = 0.0
            apply_clause_links(
                clause_internal["id"], action.action, self._clause_internals
            )
            return True

        if response.outcome == "rewritten":
            selected = self._option_by_id(
                clause_internal["rewrite_options"], action.rewrite_option_id or ""
            )
            clause_internal["status"] = "resolved"
            clause_internal["resolution"] = "rewritten"
            clause_internal["chosen_option"] = selected["id"]
            clause_internal["chosen_option_quality"] = float(selected["quality"])
            apply_clause_links(
                clause_internal["id"], action.action, self._clause_internals
            )
            return True

        return False

    def _resolve_pending_as_accepted(self) -> None:
        for clause in self._clause_internals:
            if clause["status"] == "pending":
                clause["status"] = "resolved"
                clause["resolution"] = "accepted"
                clause["chosen_option"] = None
                clause["chosen_option_quality"] = 0.0

    def _check_termination_after_step(self, message: str) -> tuple[bool, str]:
        if self._temperature <= 0.0:
            self._deal_alive = False
            return True, message

        if self._all_clauses_resolved():
            self._deal_alive = True
            return True, message

        if self._steps_taken >= self._max_steps:
            self._resolve_pending_as_accepted()
            self._deal_alive = True
            return True, "Max steps reached. Negotiation auto-finalized."

        return False, message

    def _all_clauses_resolved(self) -> bool:
        return all(clause["status"] == "resolved" for clause in self._clause_internals)

    def _update_state(self, episode_id: str | None = None) -> None:
        current_episode_id = (
            episode_id if episode_id is not None else self._state.episode_id
        )
        self._state = NegotiationState(
            episode_id=current_episode_id,
            step_count=self._steps_taken,
            task_id=self._task_id,
            contract_type=self._contract_type,
            negotiation_temperature=self._temperature,
            clauses_total=len(self._clause_internals),
            clauses_resolved=sum(
                1 for clause in self._clause_internals if clause["status"] == "resolved"
            ),
            deal_alive=self._deal_alive,
        )

    def _build_terminal_observation(self, message: str) -> NegotiationObservation:
        self._episode_done = True
        final_reward = Grader.compute_final_reward(
            clauses=self._clause_internals,
            client_priorities=self._client_priorities,
            deal_alive=self._deal_alive,
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
        )
        return self._build_observation(
            done=True, reward=final_reward, counterparty_message=message
        )

    def _build_observation(
        self,
        done: bool,
        reward: float | None,
        counterparty_message: str,
    ) -> NegotiationObservation:
        return NegotiationObservation(
            done=done,
            reward=reward,
            contract_type=self._contract_type,
            deal_context=self._deal_context,
            clauses=self._build_clause_views(),
            counterparty_message=counterparty_message,
            negotiation_temperature=self._temperature,
            steps_taken=self._steps_taken,
            max_steps=self._max_steps,
            task_id=self._task_id,
        )

    def _build_clause_views(self) -> list[ClauseView]:
        views: list[ClauseView] = []
        for clause in self._clause_internals:
            rewrite_options = []
            if clause["status"] == "pending":
                rewrite_options = [
                    RewriteOptionView(
                        id=option["id"], description=option["description"]
                    )
                    for option in clause["rewrite_options"]
                ]

            views.append(
                ClauseView(
                    index=clause["index"],
                    category=clause["category"],
                    text=clause["text"],
                    status=clause["status"],
                    rewrite_options=rewrite_options,
                    resolution=clause["resolution"],
                    chosen_option=clause["chosen_option"],
                )
            )
        return views

    def _option_by_id(self, options: list[dict], option_id: str) -> dict:
        for option in options:
            if option["id"] == option_id:
                return option
        raise ValueError(f"Unknown rewrite option: {option_id}")

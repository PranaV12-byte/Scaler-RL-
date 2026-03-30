from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import NegotiationAction, NegotiationObservation, NegotiationState


class ContractNegotiationClient(
    EnvClient[NegotiationAction, NegotiationObservation, NegotiationState]
):
    def _step_payload(self, action: NegotiationAction) -> dict:
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: dict) -> StepResult[NegotiationObservation]:
        return StepResult(
            observation=NegotiationObservation.model_validate(payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> NegotiationState:
        return NegotiationState.model_validate(payload)

from typing import Optional

from pydantic import BaseModel

from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)


class RewriteOptionView(BaseModel):
    id: str
    description: str


class ClauseView(BaseModel):
    index: int
    category: str
    text: str
    status: str
    rewrite_options: list[RewriteOptionView] = []
    resolution: Optional[str] = None
    chosen_option: Optional[str] = None


class DealContext(BaseModel):
    deal_value: str
    client_industry: str
    client_priorities: list[str]
    counterparty_name: str


class NegotiationObservation(BaseObservation):
    contract_type: str
    deal_context: DealContext
    clauses: list[ClauseView]
    counterparty_message: str
    negotiation_temperature: float
    steps_taken: int
    max_steps: int
    task_id: str


class NegotiationAction(BaseAction):
    clause_index: Optional[int] = None
    action: str
    rewrite_option_id: Optional[str] = None


class NegotiationState(BaseState):
    task_id: str = ""
    contract_type: str = ""
    negotiation_temperature: float = 1.0
    clauses_total: int = 0
    clauses_resolved: int = 0
    deal_alive: bool = True

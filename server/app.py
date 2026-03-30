import uvicorn

from openenv.core.env_server.http_server import create_app

from models import NegotiationAction, NegotiationObservation
from server.environment import ContractNegotiationEnv


app = create_app(
    ContractNegotiationEnv,
    NegotiationAction,
    NegotiationObservation,
    env_name="contract_negotiation",
)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

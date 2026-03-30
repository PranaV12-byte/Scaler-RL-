import re

import pytest

from server.contract_generator import ContractGenerator


def test_generate_easy_saas_returns_8_clauses() -> None:
    generator = ContractGenerator()
    contract = generator.generate(task_id="easy_saas", seed=42)
    assert contract.contract_type == "saas_vendor"
    assert len(contract.clauses) == 8


def test_generate_medium_freelancer_returns_10_clauses() -> None:
    generator = ContractGenerator()
    contract = generator.generate(task_id="medium_freelancer", seed=42)
    assert contract.contract_type == "freelancer"
    assert len(contract.clauses) == 10


def test_generate_hard_lease_returns_12_clauses() -> None:
    generator = ContractGenerator()
    contract = generator.generate(task_id="hard_lease", seed=42)
    assert contract.contract_type == "commercial_lease"
    assert len(contract.clauses) == 12


def test_same_seed_and_task_id_is_reproducible() -> None:
    generator = ContractGenerator()
    first = generator.generate(task_id="easy_saas", seed=42)
    second = generator.generate(task_id="easy_saas", seed=42)
    assert first == second


def test_different_seeds_change_parameterization() -> None:
    generator = ContractGenerator()
    first = generator.generate(task_id="easy_saas", seed=42)
    second = generator.generate(task_id="easy_saas", seed=43)

    first_texts = [clause.text for clause in first.clauses]
    second_texts = [clause.text for clause in second.clauses]
    assert first_texts != second_texts


def test_generated_text_has_no_unfilled_placeholders() -> None:
    generator = ContractGenerator()
    contract = generator.generate(task_id="hard_lease", seed=42)
    placeholder_pattern = re.compile(r"\{[^}]+\}")

    assert all(
        not placeholder_pattern.search(clause.text) for clause in contract.clauses
    )


def test_deal_context_is_generated() -> None:
    generator = ContractGenerator()
    contract = generator.generate(task_id="easy_saas", seed=42)

    assert contract.deal_context.deal_value
    assert contract.deal_context.client_industry
    assert len(contract.deal_context.client_priorities) == 2
    assert contract.deal_context.counterparty_name


def test_unknown_task_id_raises_value_error() -> None:
    generator = ContractGenerator()
    with pytest.raises(ValueError):
        generator.generate(task_id="unknown_task", seed=42)

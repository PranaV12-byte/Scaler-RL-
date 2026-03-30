from __future__ import annotations


class Grader:
    @staticmethod
    def compute_step_reward(
        clause_category: str,
        client_priorities: list[str],
        action_wasted: bool,
        was_refused: bool = False,
        clause_resolved: bool = True,
    ) -> float:
        if action_wasted:
            return -0.01
        if was_refused:
            return 0.0
        if not clause_resolved:
            return 0.0
        if clause_category in client_priorities:
            return 0.02
        return 0.01

    @classmethod
    def compute_final_reward(
        cls,
        clauses: list[dict],
        client_priorities: list[str],
        deal_alive: bool,
        steps_taken: int,
        max_steps: int,
    ) -> float:
        if not deal_alive:
            return 0.0

        risk_reduction = cls.compute_risk_reduction(clauses)
        efficiency = cls.compute_efficiency(steps_taken, max_steps)
        priority_alignment = cls.compute_priority_alignment(clauses, client_priorities)

        score = risk_reduction * 0.6 + efficiency * 0.15 + priority_alignment * 0.25
        return cls._clamp(score)

    @classmethod
    def compute_risk_reduction(cls, clauses: list[dict]) -> float:
        if not clauses:
            return 0.0
        values = [cls._clause_reduction(clause) for clause in clauses]
        return sum(values) / len(values)

    @staticmethod
    def compute_efficiency(steps_taken: int, max_steps: int) -> float:
        if max_steps <= 0:
            return 0.0
        return max(0.0, 1.0 - (steps_taken / max_steps))

    @classmethod
    def compute_priority_alignment(
        cls,
        clauses: list[dict],
        client_priorities: list[str],
    ) -> float:
        if not clauses:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0
        for clause in clauses:
            weight = 2 if clause["category"] in client_priorities else 1
            weighted_sum += cls._clause_reduction(clause) * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    @staticmethod
    def _clause_reduction(clause: dict) -> float:
        resolution = clause.get("resolution", "accepted")
        risk_level = float(clause.get("risk_level", 0.0))
        if resolution == "accepted":
            return 0.0
        if resolution == "rejected":
            return risk_level
        if resolution == "rewritten":
            quality = float(clause.get("chosen_option_quality", 0.0))
            return risk_level * quality
        return 0.0

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

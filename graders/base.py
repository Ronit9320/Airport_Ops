from abc import ABC, abstractmethod
from typing import Any


class BaseGrader:
    def __init__(self):
        self.actions: list[dict] = []
        self.state_history: list[dict] = []

    def record_action(self, action: dict, state: dict) -> None:
        self.actions.append(action)
        self.state_history.append(state)

    def reset(self) -> None:
        self.actions = []
        self.state_history = []

    @abstractmethod
    def grade(self) -> float:
        pass

    @abstractmethod
    def get_breakdown(self) -> dict[str, float]:
        pass

    def check_priority_ordering(self, expected_order: list[str]) -> float:
        if not self.actions:
            return 0.0
        action_order = [
            a["flight_id"]
            for a in self.actions
            if a.get("action_type") == "assign_runway"
        ]
        if not action_order:
            return 0.0
        correct = sum(
            1 for i, fid in enumerate(action_order) if fid == expected_order[i]
        )
        return correct / len(expected_order) if expected_order else 0.0

    def check_gate_types(self, gate_assignments: dict[str, str]) -> float:
        if not self.actions:
            return 0.0
        correct = 0
        total = len(gate_assignments)
        for a in self.actions:
            if a.get("action_type") == "assign_gate":
                fid = a.get("flight_id")
                target = a.get("target_id")
                if fid in gate_assignments and gate_assignments[fid] == target:
                    correct += 1
        return correct / total if total > 0 else 0.0

    def compute_penalty(self, hard_violation: bool = False) -> float:
        if hard_violation:
            return 1.0
        return 0.0

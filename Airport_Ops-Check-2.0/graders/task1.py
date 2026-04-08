from graders.base import BaseGrader


class Task1Grader(BaseGrader):
    def __init__(self):
        super().__init__()
        self.medevac_landed = False
        self.medevac_step = None
        self.crisis_step = 0

    def grade(self) -> float:
        breakdown = self.get_breakdown()
        order_score = breakdown["priority_order"]
        medevac_score = breakdown["medevac_response"]
        gate_score = breakdown["gate_types"]

        score = 0.4 * order_score + 0.4 * medevac_score + 0.2 * gate_score
        return max(0.0, min(1.0, score))

    def get_breakdown(self) -> dict[str, float]:
        order_score = 0.0
        medevac_score = 0.0
        gate_score = 0.0

        runway_actions = [
            a for a in self.actions if a.get("action_type") == "assign_runway"
        ]
        expected_order = ["FL005", "FL001", "FL002", "FL003", "FL004"]

        for i, action in enumerate(runway_actions):
            if i < len(expected_order) and action.get("flight_id") == expected_order[i]:
                order_score += 0.2
            if action.get("flight_id") == "FL005":
                self.medevac_landed = True

        if self.medevac_landed:
            medevac_score = 1.0

        gate_actions = [
            a for a in self.actions if a.get("action_type") == "assign_gate"
        ]
        expected_gates = {
            "FL005": "gate_M1",
            "FL001": "gate_A1",
            "FL002": "gate_A2",
            "FL003": "gate_B1",
            "FL004": "gate_B2",
        }
        gate_correct = sum(
            1
            for a in gate_actions
            if expected_gates.get(a.get("flight_id")) == a.get("target_id")
        )
        if expected_gates:
            gate_score = gate_correct / len(expected_gates)

        return {
            "priority_order": order_score,
            "medevac_response": medevac_score,
            "gate_types": gate_score,
        }

    def check_hard_penalties(self) -> tuple[float, bool]:
        for action in self.actions:
            if (
                action.get("flight_id") == "FL005"
                and action.get("action_type") == "assign_gate"
            ):
                target = action.get("target_id", "")
                if target != "gate_M1":
                    return 1.0, True
        return 0.0, False

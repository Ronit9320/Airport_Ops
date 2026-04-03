from graders.base import BaseGrader


class Task2Grader(BaseGrader):
    def __init__(self):
        super().__init__()
        self.fuel_flight_landed = False
        self.army_flight_landed = False
        self.isolation_used = False

    def grade(self) -> float:
        breakdown = self.get_breakdown()
        fuel_override = breakdown["fuel_override"]
        bomb_protocol = breakdown["bomb_protocol"]
        maintenance_penalty = breakdown["maintenance_check"]
        order_score = breakdown["priority_order"]

        score = (
            0.3 * fuel_override
            + 0.3 * bomb_protocol
            + 0.2 * maintenance_penalty
            + 0.2 * order_score
        )
        return max(0.0, min(1.0, score))

    def get_breakdown(self) -> dict[str, float]:
        fuel_override = 0.0
        bomb_protocol = 0.0
        maintenance_check = 1.0
        priority_order = 0.0

        runway_actions = [
            a for a in self.actions if a.get("action_type") == "assign_runway"
        ]
        gate_actions = [
            a for a in self.actions if a.get("action_type") == "assign_gate"
        ]

        fuel_landed_step = None
        army_landed_step = None
        for i, action in enumerate(runway_actions):
            if action.get("flight_id") == "FL002":
                self.fuel_flight_landed = True
                fuel_landed_step = i
            if action.get("flight_id") == "FL001":
                self.army_flight_landed = True
                army_landed_step = i

        if fuel_landed_step is not None and army_landed_step is not None:
            if fuel_landed_step < army_landed_step:
                fuel_override = 1.0
        elif fuel_landed_step is not None:
            fuel_override = 1.0

        for action in gate_actions:
            if action.get("flight_id") == "FL006":
                if action.get("target_id") == "gate_I1":
                    self.isolation_used = True
                    bomb_protocol = 1.0
                else:
                    return {
                        "fuel_override": 0.0,
                        "bomb_protocol": 0.0,
                        "maintenance_check": 1.0,
                        "priority_order": 0.0,
                    }

        for action in runway_actions:
            if action.get("target_id") == "runway1":
                maintenance_check = 0.0

        remaining_flights = ["FL003", "FL004", "FL005", "FL007", "FL008"]
        action_fids = [
            a.get("flight_id")
            for a in runway_actions
            if a.get("flight_id") in remaining_flights
        ]
        correct = sum(1 for fid in action_fids if fid in remaining_flights)
        priority_order = correct / len(remaining_flights) if remaining_flights else 0.0

        return {
            "fuel_override": fuel_override,
            "bomb_protocol": bomb_protocol,
            "maintenance_check": maintenance_check,
            "priority_order": priority_order,
        }

    def check_hard_penalties(self) -> tuple[float, bool]:
        for action in self.actions:
            if (
                action.get("flight_id") == "FL006"
                and action.get("action_type") == "assign_gate"
            ):
                target = action.get("target_id", "")
                if "isolation" not in target:
                    return 1.0, True
        return 0.0, False

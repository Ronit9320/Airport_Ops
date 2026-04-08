from graders.base import BaseGrader


class Task3Grader(BaseGrader):
    def __init__(self):
        super().__init__()
        self.secure_channel_used = False
        self.remote_stand_used = False
        self.go_around_issued = False
        self.fire_scrambled = False
        self.runway_closed = False
        self.cross_crisis_interference = False

    def grade(self) -> float:
        breakdown = self.get_breakdown()
        hijack_score = breakdown["hijack_protocol"]
        fire_score = breakdown["fire_protocol"]
        isolation_score = breakdown["crisis_isolation"]
        throughput_score = breakdown["throughput"]
        cargo_score = breakdown["cargo_handling"]

        score = (
            0.25 * hijack_score
            + 0.25 * fire_score
            + 0.20 * isolation_score
            + 0.15 * throughput_score
            + 0.15 * cargo_score
        )
        return max(0.0, min(1.0, score))

    def get_breakdown(self) -> dict[str, float]:
        hijack_protocol = 0.0
        fire_protocol = 0.0
        crisis_isolation = 1.0
        throughput = 0.0
        cargo_handling = 0.0

        hijack_actions = [a for a in self.actions if a.get("flight_id") == "FL014"]
        fire_actions = [a for a in self.actions if a.get("flight_id") == "FL005"]

        if hijack_actions:
            if any(a.get("use_secure_channel") for a in hijack_actions):
                self.secure_channel_used = True
            if any(
                a.get("action_type") == "assign_gate"
                and "isolation" in str(a.get("target_id", ""))
                for a in hijack_actions
            ):
                self.remote_stand_used = True
            if any(a.get("action_type") == "scramble_security" for a in hijack_actions):
                pass

        if self.secure_channel_used and self.remote_stand_used:
            hijack_protocol = 1.0

        if fire_actions:
            if any(a.get("action_type") == "hold" for a in fire_actions):
                self.go_around_issued = True
            if any(a.get("action_type") == "scramble_fire" for a in fire_actions):
                self.fire_scrambled = True
            if any(a.get("action_type") == "close_runway" for a in fire_actions):
                self.runway_closed = True

        if self.go_around_issued and self.fire_scrambled and self.runway_closed:
            fire_protocol = 1.0

        non_crisis_actions = [
            a for a in self.actions if a.get("flight_id") not in ["FL014", "FL005"]
        ]
        throughput = len(non_crisis_actions) / 13.0

        cargo_actions = [
            a for a in self.actions if a.get("flight_id") in ["FL013", "FL015"]
        ]
        cargo_to_cargo_gate = sum(
            1
            for a in cargo_actions
            if a.get("action_type") == "assign_gate"
            and "cargo" in str(a.get("target_id", ""))
        )
        if cargo_actions:
            cargo_handling = cargo_to_cargo_gate / len(cargo_actions)

        return {
            "hijack_protocol": hijack_protocol,
            "fire_protocol": fire_protocol,
            "crisis_isolation": crisis_isolation,
            "throughput": throughput,
            "cargo_handling": cargo_handling,
        }

    def check_hard_penalties(self) -> tuple[float, bool]:
        for action in self.actions:
            if (
                action.get("flight_id") == "FL014"
                and action.get("action_type") == "assign_gate"
            ):
                target = action.get("target_id", "")
                # Check if it's isolation gate (either contains "isolation" or is gate_I1)
                is_isolation = "isolation" in target.lower() or target in ("gate_I1", "gate_ISO1")
                if not is_isolation:
                    return 1.0, True
                if not action.get("use_secure_channel"):
                    return 1.0, True
        return 0.0, False

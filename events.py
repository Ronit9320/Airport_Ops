from typing import Optional
from state import AirportState, CrisisEvent, Flight
import json


class CrisisProtocol:
    MEDICAL_REQUIREMENTS = ["clear_runway", "medical_gate", "scramble_ambulance"]
    BOMB_REQUIREMENTS = [
        "isolation_bay",
        "halt_movement",
        "notify_security",
        "notify_fire",
        "notify_police",
    ]
    HIJACK_REQUIREMENTS = ["remote_stand", "secure_channel", "scramble_security"]
    FIRE_REQUIREMENTS = [
        "issue_go_around",
        "scramble_fire",
        "close_runway",
        "reroute_landings",
    ]
    FUEL_REQUIREMENTS = ["immediate_runway"]

    @staticmethod
    def get_requirements(crisis_type: str) -> list[str]:
        mapping = {
            "medical_emergency": CrisisProtocol.MEDICAL_REQUIREMENTS,
            "bomb_threat": CrisisProtocol.BOMB_REQUIREMENTS,
            "hijacking": CrisisProtocol.HIJACK_REQUIREMENTS,
            "runway_fire": CrisisProtocol.FIRE_REQUIREMENTS,
            "fuel_emergency": CrisisProtocol.FUEL_REQUIREMENTS,
        }
        return mapping.get(crisis_type, [])


class EventManager:
    def __init__(self, state: AirportState, crisis_events: list[CrisisEvent]):
        self.state = state
        self.crisis_events = crisis_events
        self._action_log: list[dict] = []

    def get_active_crises(self, current_step: int) -> list[CrisisEvent]:
        active = []
        for crisis in self.crisis_events:
            if crisis.activate_step <= current_step and not crisis.resolved:
                active.append(crisis)
        return active

    def check_protocol_compliance(
        self, action: dict, crisis: CrisisEvent
    ) -> tuple[float, list[str]]:
        requirements = CrisisProtocol.get_requirements(crisis.type)
        met = []
        missing = []
        score = 1.0

        if crisis.type == "medical_emergency":
            if action.get("action_type") == "assign_runway":
                met.append("clear_runway")
            if action.get("action_type") == "assign_gate" and "medical" in str(
                action.get("target_id", "")
            ):
                met.append("medical_gate")
            if action.get("action_type") == "scramble_medical":
                met.append("scramble_ambulance")

        elif crisis.type == "bomb_threat":
            if action.get("action_type") == "assign_gate" and "isolation" in str(
                action.get("target_id", "")
            ):
                met.append("isolation_bay")
            if action.get("action_type") == "hold":
                met.append("halt_movement")
            if "notify_security" in action:
                met.append("notify_security")
                met.append("notify_fire")
                met.append("notify_police")

        elif crisis.type == "hijacking":
            if action.get("action_type") == "assign_gate" and "isolation" in str(
                action.get("target_id", "")
            ):
                met.append("remote_stand")
            if action.get("use_secure_channel"):
                met.append("secure_channel")
            if action.get("action_type") == "scramble_security":
                met.append("scramble_security")

        elif crisis.type == "runway_fire":
            if action.get("action_type") == "hold":
                met.append("issue_go_around")
            if action.get("action_type") == "scramble_fire":
                met.append("scramble_fire")
            if action.get("close_runway"):
                met.append("close_runway")

        elif crisis.type == "fuel_emergency":
            if action.get("action_type") == "assign_runway":
                met.append("immediate_runway")

        missing = [r for r in requirements if r not in met]
        if missing:
            score = len(met) / len(requirements)

        return score, missing

    def validate_action(self, action: dict) -> tuple[bool, str]:
        flight_id = action.get("flight_id")
        action_type = action.get("action_type")
        target_id = action.get("target_id")

        if flight_id and flight_id not in self.state.flights:
            return False, f"Unknown flight: {flight_id}"

        if action_type == "assign_runway" and target_id:
            if target_id not in self.state.runways:
                return False, f"Unknown runway: {target_id}"
            runway = self.state.runways[target_id]
            if runway.status != "free":
                return False, f"Runway {target_id} is not available"

        if action_type == "assign_gate" and target_id:
            if target_id not in self.state.gates:
                return False, f"Unknown gate: {target_id}"
            gate = self.state.gates[target_id]
            if gate.status != "free":
                return False, f"Gate {target_id} is not available"

        return True, ""

    def log_action(self, action: dict) -> None:
        self._action_log.append(action.copy())

    def get_action_log(self) -> list[dict]:
        return self._action_log.copy()

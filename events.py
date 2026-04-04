from typing import Optional
from state import AirportState, CrisisEvent, Flight
 
 
class CrisisProtocol:
    REQUIREMENTS = {
        "medical_emergency": ["clear_runway", "medical_gate", "scramble_ambulance"],
        "bomb_threat":       ["isolation_bay", "halt_movement", "notify_security"],
        "hijacking":         ["isolation_gate", "secure_channel", "scramble_security"],
        "runway_fire":       ["issue_go_around", "scramble_fire", "close_runway"],
        "fuel_emergency":    ["immediate_runway"],
    }
 
    @staticmethod
    def get_requirements(crisis_type: str) -> list[str]:
        return CrisisProtocol.REQUIREMENTS.get(crisis_type, [])
 
 
class EventManager:
    def __init__(self, state: AirportState, crisis_events: list[CrisisEvent]):
        self.state = state
        self.crisis_events = crisis_events
        self._action_log: list[dict] = []
        # Track what protocol steps have been satisfied per crisis flight_id
        self._protocol_progress: dict[str, set] = {}
 
    def get_active_crises(self, current_step: int) -> list[CrisisEvent]:
        return [
            c for c in self.crisis_events
            if c.activate_step <= current_step and not c.resolved
        ]
 
    def check_protocol_compliance(
        self, action: dict, crisis: CrisisEvent
    ) -> tuple[float, list[str]]:
        """
        Score how well this single action satisfies protocol requirements for the crisis.
        Updates internal progress tracker so multi-step protocols accumulate.
        """
        requirements = CrisisProtocol.get_requirements(crisis.type)
        fid = crisis.flight_id
 
        if fid not in self._protocol_progress:
            self._protocol_progress[fid] = set()
 
        progress = self._protocol_progress[fid]
 
        if crisis.type == "medical_emergency":
            if action.get("action_type") == "assign_runway":
                progress.add("clear_runway")
            if (
                action.get("action_type") == "assign_gate"
                and "med" in str(action.get("target_id", "")).lower()
            ):
                progress.add("medical_gate")
            # Also accept gate type lookup
            if action.get("action_type") == "assign_gate":
                gate_id = action.get("target_id", "")
                gate = self.state.gates.get(gate_id)
                if gate and gate.type.value == "medical":
                    progress.add("medical_gate")
            if action.get("action_type") == "scramble_medical":
                progress.add("scramble_ambulance")
 
        elif crisis.type == "bomb_threat":
            if action.get("action_type") == "assign_gate":
                gate_id = action.get("target_id", "")
                gate = self.state.gates.get(gate_id)
                if gate and gate.type.value == "isolation":
                    progress.add("isolation_bay")
            if action.get("action_type") == "hold":
                progress.add("halt_movement")
            notify = action.get("notify_authorities", []) or []
            if "security" in notify:
                progress.add("notify_security")
 
        elif crisis.type == "hijacking":
            if action.get("action_type") == "assign_gate":
                gate_id = action.get("target_id", "")
                gate = self.state.gates.get(gate_id)
                if gate and gate.type.value == "isolation":
                    progress.add("isolation_gate")
            if action.get("use_secure_channel"):
                progress.add("secure_channel")
            if action.get("action_type") == "scramble_security":
                progress.add("scramble_security")
 
        elif crisis.type == "runway_fire":
            if action.get("action_type") == "hold":
                progress.add("issue_go_around")
            if action.get("action_type") == "scramble_fire":
                progress.add("scramble_fire")
            if action.get("action_type") == "close_runway":
                progress.add("close_runway")
 
        elif crisis.type == "fuel_emergency":
            if action.get("flight_id") == fid and action.get("action_type") == "assign_runway":
                progress.add("immediate_runway")
 
        missing = [r for r in requirements if r not in progress]
        score = len(progress.intersection(set(requirements))) / len(requirements) if requirements else 1.0
        return round(score, 4), missing
 
    def check_full_protocol_completion(self, crisis: CrisisEvent) -> tuple[float, list[str]]:
        """Check overall completion across all accumulated actions for a crisis."""
        requirements = CrisisProtocol.get_requirements(crisis.type)
        progress = self._protocol_progress.get(crisis.flight_id, set())
        missing = [r for r in requirements if r not in progress]
        score = len(progress.intersection(set(requirements))) / len(requirements) if requirements else 1.0
        return round(score, 4), missing
 
    def validate_action(self, action: dict) -> tuple[bool, str]:
        flight_id = action.get("flight_id")
        action_type = action.get("action_type")
        target_id = action.get("target_id")
 
        if flight_id and flight_id not in self.state.flights:
            return False, f"Unknown flight: {flight_id}"
 
        if action_type == "assign_runway" and target_id:
            if target_id not in self.state.runways:
                return False, f"Unknown runway: {target_id}"
            from state import RunwayStatus
            runway = self.state.runways[target_id]
            if runway.status == RunwayStatus.MAINTENANCE:
                return False, f"Runway {target_id} is under maintenance"
            if runway.status == RunwayStatus.CLOSED:
                return False, f"Runway {target_id} is closed"
            if runway.status != RunwayStatus.FREE:
                return False, f"Runway {target_id} is not available (status: {runway.status})"
 
        if action_type == "assign_gate" and target_id:
            if target_id not in self.state.gates:
                return False, f"Unknown gate: {target_id}"
            from state import GateStatus
            gate = self.state.gates[target_id]
            if gate.status != GateStatus.FREE:
                return False, f"Gate {target_id} is not available (status: {gate.status})"
 
        if action_type == "close_runway" and target_id:
            if target_id not in self.state.runways:
                return False, f"Unknown runway: {target_id}"
 
        return True, ""
 
    def log_action(self, action: dict) -> None:
        self._action_log.append(action.copy())
 
    def get_action_log(self) -> list[dict]:
        return self._action_log.copy()
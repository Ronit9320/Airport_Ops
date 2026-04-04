from typing import Optional
from models import Observation, Action, Reward
from state import AirportStateMachine, RunwayStatus, GateStatus, GateType
from events import EventManager
from graders import Task1Grader, Task2Grader, Task3Grader
from graders.base import BaseGrader
 
 
class AirportOpsEnv:
    def __init__(self):
        self.state_machine: Optional[AirportStateMachine] = None
        self.event_manager: Optional[EventManager] = None
        self.grader: Optional[BaseGrader] = None
        self.current_task: Optional[str] = None
        self.max_steps: int = 20
        self.done: bool = False
 
    def reset(self, task_id: str = "task1") -> Observation:
        scenario_path = f"data/scenarios/{task_id}.json"
        self.state_machine = AirportStateMachine(scenario_path)
        self.event_manager = EventManager(
            self.state_machine.state, self.state_machine.state.active_crises
        )
        self.current_task = task_id
 
        task_config = {
            "task1": (Task1Grader, 20),
            "task2": (Task2Grader, 40),
            "task3": (Task3Grader, 80),
        }
        grader_cls, max_steps = task_config.get(task_id, (Task1Grader, 20))
        self.grader = grader_cls()
        self.max_steps = max_steps
        self.done = False
        return self._get_observation()
 
    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            return (
                self._get_observation(),
                self._get_zero_reward(),
                True,
                {"error": "Episode already ended. Call reset()."},
            )
 
        action_dict = action.model_dump()
 
        # Validate before applying
        valid, error_msg = self.event_manager.validate_action(action_dict)
        if not valid:
            return (
                self._get_observation(),
                self._get_zero_reward(),
                False,
                {"error": error_msg},
            )
 
        # Apply the action to state
        self._apply_action(action_dict)
        self.event_manager.log_action(action_dict)
 
        # Record for grader AFTER applying (so state reflects the action's result)
        self.state_machine.increment_step()
 
        if self.grader:
            self.grader.record_action(action_dict, self.state_machine.to_dict())
 
        # Auto-resolve crises whose protocol is complete
        self._auto_resolve_crises()
 
        reward = self._compute_reward(action_dict)
        self.done = self._check_done()
 
        return self._get_observation(), reward, self.done, {}
 
    def state(self) -> dict:
        if self.state_machine:
            s = self.state_machine.to_dict()
            s.pop("state_history", None)  # don't bloat the HTTP response
            return s
        return {}
 
    def _apply_action(self, action: dict) -> None:
        flight_id = action.get("flight_id")
        action_type = action.get("action_type")
        target_id = action.get("target_id")
 
        if action_type == "assign_runway" and target_id:
            self.state_machine.assign_runway(flight_id, target_id)
 
        elif action_type == "assign_gate" and target_id:
            self.state_machine.assign_gate(flight_id, target_id)
 
        elif action_type == "hold":
            self.state_machine.hold_flight(flight_id)
 
        elif action_type == "divert":
            self.state_machine.divert_flight(flight_id)
 
        elif action_type == "scramble_security":
            self.state_machine.scramble_unit("security")
 
        elif action_type == "scramble_fire":
            self.state_machine.scramble_unit("fire_truck")
 
        elif action_type == "scramble_medical":
            self.state_machine.scramble_unit("ambulance")
 
        elif action_type == "close_runway" and target_id:
            self.state_machine.close_runway(target_id)
 
        elif action_type == "vacate_runway" and target_id:
            self.state_machine.vacate_runway(target_id)
 
    def _auto_resolve_crises(self) -> None:
        """Mark crisis as resolved if all required protocol steps have been taken."""
        active_crises = self.event_manager.get_active_crises(
            self.state_machine.state.step
        )
        for crisis in active_crises:
            score, missing = self.event_manager.check_full_protocol_completion(crisis)
            if score >= 1.0 and not missing:
                crisis.resolved = True
 
    def _compute_reward(self, action: dict) -> Reward:
        if not self.grader:
            return self._get_zero_reward()
 
        hard_penalty, is_hard = self.grader.check_hard_penalties()
        if is_hard:
            return Reward(
                total=0.0,
                priority_score=0.0,
                resource_match_score=0.0,
                eta_score=0.0,
                crisis_protocol_score=0.0,
                penalty=1.0,
            )
 
        priority_score = self._score_priority(action)
        resource_score = self._score_resource_match(action)
        eta_score = self._score_eta(action)
        crisis_score = self._score_crisis_protocol(action)
        penalty = hard_penalty
 
        total = (
            0.30 * priority_score
            + 0.25 * resource_score
            + 0.20 * eta_score
            + 0.15 * crisis_score
            + 0.10 * (1.0 - penalty)
        )
 
        if self.grader._used_maintenance_runway():
            total = min(total, 0.2)
 
        return Reward(
            total=round(max(0.0, min(1.0, total)), 4),
            priority_score=round(priority_score, 4),
            resource_match_score=round(resource_score, 4),
            eta_score=round(eta_score, 4),
            crisis_protocol_score=round(crisis_score, 4),
            penalty=round(penalty, 4),
        )
 
    def _score_priority(self, action: dict) -> float:
        """Score whether the chosen flight was the highest priority waiting."""
        if action.get("action_type") not in ["assign_runway", "assign_gate"]:
            return 1.0
        state = self.state_machine.state
        flight_id = action.get("flight_id")
        if not flight_id:
            return 0.5
 
        waiting = [
            fid for fid, f in state.flights.items()
            if f.status in ["requesting_landing", "holding"]
        ]
        if not waiting:
            return 1.0
 
        optimal = min(waiting, key=lambda fid: state.get_flight_priority(fid))
        chosen_priority = state.get_flight_priority(flight_id)
        optimal_priority = state.get_flight_priority(optimal)
 
        if chosen_priority == optimal_priority:
            return 1.0
        # Partial score: penalise proportionally to how far off priority is
        max_priority = 5
        gap = abs(chosen_priority - optimal_priority)
        return max(0.0, 1.0 - gap * 0.2)
 
    def _score_resource_match(self, action: dict) -> float:
        """Score gate/runway type suitability for the flight."""
        flight_id = action.get("flight_id")
        target_id = action.get("target_id")
        action_type = action.get("action_type")
 
        if not flight_id or not target_id or action_type not in ["assign_gate", "assign_runway"]:
            return 1.0
 
        state = self.state_machine.state
        flight = state.flights.get(flight_id)
        if not flight:
            return 1.0
 
        if action_type == "assign_gate":
            gate = state.gates.get(target_id)
            if not gate:
                return 0.0
            gtype = gate.type
            crisis = flight.crisis
            ftype = flight.flight_type
 
            if crisis in ("hijack",) or crisis == "bomb_threat":
                return 1.0 if gtype == GateType.ISOLATION else 0.0
            if ftype == "medevac" or crisis == "medical_onboard":
                return 1.0 if gtype == GateType.MEDICAL else 0.0
            if ftype == "cargo":
                return 1.0 if gtype in [GateType.CARGO, GateType.ISOLATION] else 0.2
            # commercial / government / army → pax gate
            return 1.0 if gtype == GateType.PAX else 0.5
 
        if action_type == "assign_runway":
            runway = state.runways.get(target_id)
            if not runway:
                return 0.0
            from state import RunwayStatus
            if runway.status in (RunwayStatus.MAINTENANCE, RunwayStatus.CLOSED):
                return 0.0
            return 1.0
 
        return 1.0
 
    def _score_eta(self, action: dict) -> float:
        """Score ETA optimality using the real ETA lookup table."""
        if action.get("action_type") != "assign_runway":
            return 1.0  # ETA only scored on runway assignment (the landing decision)
 
        flight_id = action.get("flight_id")
        runway_id = action.get("target_id")
        if not flight_id or not runway_id:
            return 1.0
 
        state = self.state_machine.state
        flight = state.flights.get(flight_id)
        if not flight:
            return 1.0
 
        # Find the gate the agent is likely heading toward (assigned or type-matched first available)
        gate_id = flight.assigned_gate
        if not gate_id:
            # Use the closest appropriate gate type as reference
            if flight.crisis in ("hijack", "bomb_threat"):
                target_type = GateType.ISOLATION
            elif flight.flight_type == "medevac" or flight.crisis == "medical_onboard":
                target_type = GateType.MEDICAL
            elif flight.flight_type == "cargo":
                target_type = GateType.CARGO
            else:
                target_type = GateType.PAX
            available = state.get_available_gates(target_type)
            gate_id = available[0] if available else None
 
        if not gate_id:
            return 1.0
 
        return self.state_machine.score_eta_optimality(
            runway_id, gate_id, flight.flight_type
        )
 
    def _score_crisis_protocol(self, action: dict) -> float:
        active_crises = self.event_manager.get_active_crises(
            self.state_machine.state.step
        )
        if not active_crises:
            return 1.0
 
        scores = []
        for crisis in active_crises:
            score, _ = self.event_manager.check_protocol_compliance(action, crisis)
            scores.append(score)
        return max(scores) if scores else 1.0
 
    def _get_observation(self) -> Observation:
        from models import FlightInfo as FlightInfoModel
 
        state = self.state_machine.state
        active_crises = self.event_manager.get_active_crises(state.step)
 
        flights_data = [
            FlightInfoModel(
                flight_id=f.flight_id,
                flight_type=f.flight_type,
                status=f.status,
                fuel_remaining_mins=f.fuel_remaining_mins,
                passengers=f.passengers,
                crisis=f.crisis,
            )
            for f in state.flights.values()
        ]
 
        return Observation(
            step=state.step,
            time_of_day=state.time_context.time_of_day,
            day_of_week=state.time_context.day_of_week,
            is_holiday=state.time_context.is_holiday,
            flights=flights_data,
            runways={k: v.model_dump() for k, v in state.runways.items()},
            gates={k: v.model_dump() for k, v in state.gates.items()},
            active_crises=[c.flight_id for c in active_crises],
            available_runways=state.get_available_runways(),
            available_gates=state.get_available_gates(),
            ground_units=state.ground_units.model_dump(),
        )
 
    def _get_zero_reward(self) -> Reward:
        return Reward(
            total=0.0,
            priority_score=0.0,
            resource_match_score=0.0,
            eta_score=0.0,
            crisis_protocol_score=0.0,
            penalty=1.0,
        )
 
    def _check_done(self) -> bool:
        if self.state_machine.state.step >= self.max_steps:
            return True
        all_processed = all(
            f.status in ["at_gate", "diverted"]
            for f in self.state_machine.state.flights.values()
        )
        return all_processed
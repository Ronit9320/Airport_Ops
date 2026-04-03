from typing import Optional
from models import Observation, Action, Reward
from state import AirportStateMachine, RunwayStatus, GateStatus, GateType
from events import EventManager
from graders import Task1Grader, Task2Grader, Task3Grader
from graders.base import BaseGrader
import json


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
        if task_id == "task1":
            self.grader = Task1Grader()
            self.max_steps = 20
        elif task_id == "task2":
            self.grader = Task2Grader()
            self.max_steps = 40
        elif task_id == "task3":
            self.grader = Task3Grader()
            self.max_steps = 80
        else:
            self.grader = Task1Grader()
            self.max_steps = 20

        self.done = False
        return self._get_observation()

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            return (
                self._get_observation(),
                self._get_zero_reward(),
                True,
                {"error": "Episode ended"},
            )

        action_dict = action.model_dump()
        valid, error_msg = self.event_manager.validate_action(action_dict)
        if not valid:
            return (
                self._get_observation(),
                self._get_zero_reward(),
                False,
                {"error": error_msg},
            )

        self._apply_action(action_dict)
        self.event_manager.log_action(action_dict)

        if self.grader:
            self.grader.record_action(action_dict, self.state_machine.to_dict())

        self.state_machine.increment_step()

        reward = self._compute_reward(action_dict)

        active_crises = self.event_manager.get_active_crises(
            self.state_machine.state.step
        )
        if len(active_crises) > len(
            [c for c in self.state_machine.state.active_crises if c.resolved]
        ):
            for crisis in active_crises:
                if not crisis.resolved:
                    crisis.resolved = True

        self.done = self._check_done()

        return self._get_observation(), reward, self.done, {}

    def state(self) -> dict:
        if self.state_machine:
            return self.state_machine.to_dict()
        return {}

    def _apply_action(self, action: dict) -> None:
        flight_id = action.get("flight_id")
        action_type = action.get("action_type")
        target_id = action.get("target_id")
        use_secure = action.get("use_secure_channel", False)

        if action_type == "assign_runway" and target_id:
            self.state_machine.assign_runway(flight_id, target_id)
            flight = self.state_machine.state.flights.get(flight_id)
            if flight:
                flight.status = "at_gate"

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
            + 0.10 * (1 - penalty)
        )

        return Reward(
            total=round(max(0.0, min(1.0, total)), 4),
            priority_score=round(priority_score, 4),
            resource_match_score=round(resource_score, 4),
            eta_score=round(eta_score, 4),
            crisis_protocol_score=round(crisis_score, 4),
            penalty=round(penalty, 4),
        )

    def _score_priority(self, action: dict) -> float:
        if action.get("action_type") not in ["assign_runway", "assign_gate"]:
            return 1.0
        return 1.0

    def _score_resource_match(self, action: dict) -> float:
        flight_id = action.get("flight_id")
        target_id = action.get("target_id")
        action_type = action.get("action_type")

        if (
            not flight_id
            or not target_id
            or action_type not in ["assign_gate", "assign_runway"]
        ):
            return 1.0

        flight = self.state_machine.state.flights.get(flight_id)
        if not flight:
            return 1.0

        if action_type == "assign_gate":
            gate = self.state_machine.state.gates.get(target_id)
            if gate:
                if flight.crisis == "hijack" or flight.crisis == "bomb_threat":
                    return 1.0 if gate.type == GateType.ISOLATION else 0.0
                if flight.flight_type == "cargo":
                    return (
                        1.0
                        if gate.type in [GateType.CARGO, GateType.ISOLATION]
                        else 0.0
                    )
                if (
                    flight.crisis == "medical_onboard"
                    or flight.flight_type == "medevac"
                ):
                    return 1.0 if gate.type == GateType.MEDICAL else 0.0
        return 1.0

    def _score_eta(self, action: dict) -> float:
        return 1.0

    def _score_crisis_protocol(self, action: dict) -> float:
        active_crises = self.event_manager.get_active_crises(
            self.state_machine.state.step
        )
        if not active_crises:
            return 1.0

        max_score = 0.0
        for crisis in active_crises:
            score, missing = self.event_manager.check_protocol_compliance(
                action, crisis
            )
            max_score = max(max_score, score)
        return max_score

    def _get_observation(self) -> Observation:
        from models import FlightInfo as FlightInfoModel

        state = self.state_machine.state
        active_crises = self.event_manager.get_active_crises(state.step)

        flights_data = []
        for f in state.flights.values():
            flights_data.append(
                FlightInfoModel(
                    flight_id=f.flight_id,
                    flight_type=f.flight_type,
                    status=f.status,
                    fuel_remaining_mins=f.fuel_remaining_mins,
                    passengers=f.passengers,
                    crisis=f.crisis,
                )
            )

        return Observation(
            step=state.step,
            time_of_day=state.time_context.time_of_day,
            day_of_week=state.time_context.day_of_week,
            is_holiday=state.time_context.is_holiday,
            flights=flights_data,
            runways={k: v.model_dump() for k, v in state.runways.items()},
            gates={k: v.model_dump() for k, v in state.gates.items()},
            active_crises=[c.flight_id for c in active_crises],
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
        if all_processed:
            return True
        return False

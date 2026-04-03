from typing import Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum
import json


class RunwayStatus(str, Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    CLOSED = "closed"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class GateStatus(str, Enum):
    FREE = "free"
    OCCUPIED = "occupied"
    RESERVED = "reserved"


class GateType(str, Enum):
    PAX = "pax"
    CARGO = "cargo"
    MEDICAL = "medical"
    ISOLATION = "isolation"


class RunwayType(str, Enum):
    LANDING = "landing"
    TAKEOFF = "takeoff"
    DUAL = "dual"


class Runway(BaseModel):
    id: str
    type: RunwayType
    status: RunwayStatus = RunwayStatus.FREE
    assigned_flight: Optional[str] = None


class Gate(BaseModel):
    id: str
    type: GateType
    status: GateStatus = GateStatus.FREE
    assigned_flight: Optional[str] = None


class Taxiway(BaseModel):
    id: str
    clear: bool = True
    blocked_by: Optional[str] = None


class GroundUnits(BaseModel):
    ambulances: int
    fire_trucks: int
    security_teams: int


class CrisisEvent(BaseModel):
    type: Literal[
        "medical_emergency", "bomb_threat", "hijacking", "runway_fire", "fuel_emergency"
    ]
    flight_id: str
    activate_step: int
    resolved: bool = False


class Flight(BaseModel):
    flight_id: str
    flight_type: Literal["army", "medevac", "government", "commercial", "cargo"]
    status: Literal["requesting_landing", "requesting_takeoff", "at_gate", "holding"]
    fuel_remaining_mins: int
    passengers: int
    crisis: Optional[Literal["hijack", "bomb_threat", "fire", "medical_onboard"]] = None
    current_location: Optional[str] = None
    steps_waiting: int = 0
    assigned_runway: Optional[str] = None
    assigned_gate: Optional[str] = None


class TimeContext(BaseModel):
    time_of_day: str
    day_of_week: str
    is_holiday: bool


class AirportState(BaseModel):
    step: int = 0
    time_context: TimeContext
    runways: dict[str, Runway] = Field(default_factory=dict)
    gates: dict[str, Gate] = Field(default_factory=dict)
    taxiways: dict[str, Taxiway] = Field(default_factory=dict)
    flights: dict[str, Flight] = Field(default_factory=dict)
    ground_units: GroundUnits
    active_crises: list[CrisisEvent] = Field(default_factory=list)
    state_history: list[dict] = Field(default_factory=list)

    def get_time_bucket(self) -> str:
        hour = int(self.time_context.time_of_day.split(":")[0])
        day_type = (
            "weekday"
            if self.time_context.day_of_week not in ["Saturday", "Sunday"]
            else "weekend"
        )

        if self.time_context.is_holiday:
            return "holiday"
        elif day_type == "weekend":
            return "weekend"
        elif 7 <= hour <= 10 or 17 <= hour <= 20:
            return "rush_hour_weekday"
        else:
            return "off_peak_weekday"

    def get_flight_priority(self, flight_id: str) -> int:
        if flight_id not in self.flights:
            return 6
        flight = self.flights[flight_id]

        if flight.fuel_remaining_mins < 10:
            if flight.flight_type in ["commercial", "cargo"]:
                return 1
            elif flight.flight_type == "government":
                return 2

        priority_map = {
            "army": 1,
            "medevac": 2,
            "government": 3,
            "commercial": 4,
            "cargo": 5,
        }
        return priority_map.get(flight.flight_type, 6)

    def get_available_runways(self) -> list[str]:
        return [rid for rid, r in self.runways.items() if r.status == RunwayStatus.FREE]

    def get_available_gates(self, gate_type: Optional[GateType] = None) -> list[str]:
        gates = [gid for gid, g in self.gates.items() if g.status == GateStatus.FREE]
        if gate_type:
            gates = [g for g in gates if self.gates[g].type == gate_type]
        return gates

    def get_ground_truth(self) -> dict:
        return {
            "priority_order": sorted(
                [
                    fid
                    for fid in self.flights
                    if self.flights[fid].status == "requesting_landing"
                ],
                key=lambda x: self.get_flight_priority(x),
            ),
            "active_crises": [
                c.model_dump() for c in self.active_crises if not c.resolved
            ],
        }


class AirportStateMachine:
    def __init__(self, scenario_path: str):
        with open(scenario_path, "r") as f:
            scenario = json.load(f)
        self._load_scenario(scenario)

    def _load_scenario(self, scenario: dict) -> None:
        time_ctx = TimeContext(**scenario["time_context"])
        self.state = AirportState(
            time_context=time_ctx,
            ground_units=GroundUnits(**scenario["ground_units"]),
        )

        for r in scenario["runways"]:
            status = RunwayStatus.FREE
            if r.get("status") == "maintenance":
                status = RunwayStatus.MAINTENANCE
            self.state.runways[r["id"]] = Runway(
                id=r["id"],
                type=RunwayType(r["type"]),
                status=status,
            )

        for g in scenario["gates"]:
            status = GateStatus.FREE
            if g.get("status") == "occupied":
                status = GateStatus.OCCUPIED
            self.state.gates[g["id"]] = Gate(
                id=g["id"],
                type=GateType(g["type"]),
                status=status,
            )

        self.state.taxiways = {f"tw{i}": Taxiway(id=f"tw{i}") for i in range(1, 5)}

        for f in scenario["flights"]:
            self.state.flights[f["flight_id"]] = Flight(**f)

        for c in scenario.get("crises", []):
            self.state.active_crises.append(CrisisEvent(**c))

    def assign_runway(self, flight_id: str, runway_id: str) -> bool:
        if runway_id not in self.state.runways:
            return False
        runway = self.state.runways[runway_id]
        if runway.status != RunwayStatus.FREE:
            return False
        if flight_id not in self.state.flights:
            return False

        runway.status = RunwayStatus.OCCUPIED
        runway.assigned_flight = flight_id
        self.state.flights[flight_id].assigned_runway = runway_id
        self.state.flights[flight_id].current_location = runway_id
        return True

    def vacate_runway(self, runway_id: str) -> bool:
        if runway_id not in self.state.runways:
            return False
        runway = self.state.runways[runway_id]
        if runway.assigned_flight:
            flight_id = runway.assigned_flight
            self.state.flights[flight_id].assigned_runway = None
            self.state.flights[flight_id].current_location = None
        runway.status = RunwayStatus.FREE
        runway.assigned_flight = None
        return True

    def assign_gate(self, flight_id: str, gate_id: str) -> bool:
        if gate_id not in self.state.gates:
            return False
        gate = self.state.gates[gate_id]
        if gate.status != GateStatus.FREE:
            return False
        if flight_id not in self.state.flights:
            return False

        gate.status = GateStatus.OCCUPIED
        gate.assigned_flight = flight_id
        self.state.flights[flight_id].assigned_gate = gate_id
        self.state.flights[flight_id].current_location = gate_id
        self.state.flights[flight_id].status = "at_gate"
        return True

    def hold_flight(self, flight_id: str) -> bool:
        if flight_id not in self.state.flights:
            return False
        self.state.flights[flight_id].status = "holding"
        self.state.flights[flight_id].steps_waiting += 1
        return True

    def divert_flight(self, flight_id: str) -> bool:
        if flight_id not in self.state.flights:
            return False
        self.state.flights[flight_id].status = "diverted"
        return True

    def close_runway(self, runway_id: str) -> bool:
        if runway_id not in self.state.runways:
            return False
        self.state.runways[runway_id].status = RunwayStatus.CLOSED
        return True

    def scramble_unit(
        self, unit_type: Literal["ambulance", "fire_truck", "security"]
    ) -> bool:
        if unit_type == "ambulance" and self.state.ground_units.ambulances > 0:
            self.state.ground_units.ambulances -= 1
            return True
        elif unit_type == "fire_truck" and self.state.ground_units.fire_trucks > 0:
            self.state.ground_units.fire_trucks -= 1
            return True
        elif unit_type == "security" and self.state.ground_units.security_teams > 0:
            self.state.ground_units.security_teams -= 1
            return True
        return False

    def activate_crisis(self, crisis_type: str, flight_id: str) -> bool:
        for crisis in self.state.active_crises:
            if crisis.flight_id == flight_id and not crisis.resolved:
                crisis.resolved = True
                return True
        return False

    def increment_step(self) -> None:
        self.state.step += 1
        for flight in self.state.flights.values():
            if flight.status in ["requesting_landing", "holding"]:
                flight.steps_waiting += 1
                if flight.fuel_remaining_mins > 0:
                    flight.fuel_remaining_mins -= 1
        self.state.state_history.append(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "step": self.state.step,
            "time_context": self.state.time_context.model_dump(),
            "runways": {k: v.model_dump() for k, v in self.state.runways.items()},
            "gates": {k: v.model_dump() for k, v in self.state.gates.items()},
            "flights": {k: v.model_dump() for k, v in self.state.flights.items()},
            "ground_units": self.state.ground_units.model_dump(),
            "active_crises": [c.model_dump() for c in self.state.active_crises],
        }

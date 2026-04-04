


from typing import Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum
import json
import os
 
 
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
    status: Literal[
        "requesting_landing", "requesting_takeoff", "at_gate", "holding", "diverted"
    ]
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
        try:
            hour = int(self.time_context.time_of_day.split(":")[0])
        except (ValueError, IndexError):
            hour = 12
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
        """Lower number = higher priority. Fuel emergency always wins."""
        if flight_id not in self.flights:
            return 99
        flight = self.flights[flight_id]
 
        if flight.fuel_remaining_mins < 10:
            return 0  # absolute top — above army
 
        priority_map = {
            "army": 1,
            "medevac": 2,
            "government": 3,
            "commercial": 4,
            "cargo": 5,
        }
        return priority_map.get(flight.flight_type, 6)
 
    def get_available_runways(self) -> list[str]:
        return [
            rid
            for rid, r in self.runways.items()
            if r.status == RunwayStatus.FREE
        ]
 
    def get_available_gates(self, gate_type: Optional[GateType] = None) -> list[str]:
        gates = [
            gid for gid, g in self.gates.items() if g.status == GateStatus.FREE
        ]
        if gate_type:
            gates = [g for g in gates if self.gates[g].type == gate_type]
        return gates
 
    def get_ground_truth(self) -> dict:
        requesting = [
            fid
            for fid, f in self.flights.items()
            if f.status in ["requesting_landing", "holding"]
        ]
        return {
            "priority_order": sorted(requesting, key=lambda x: self.get_flight_priority(x)),
            "active_crises": [
                c.model_dump() for c in self.active_crises if not c.resolved
            ],
        }
 
 
class AirportStateMachine:
    def __init__(self, scenario_path: str):
        with open(scenario_path, "r") as f:
            scenario = json.load(f)
        self._load_scenario(scenario)
        self._load_eta_table()
 
    def _load_eta_table(self) -> None:
        eta_path = os.path.join(os.path.dirname(__file__), "data", "eta_table.json")
        try:
            with open(eta_path, "r") as f:
                self._eta_table = json.load(f)
        except FileNotFoundError:
            self._eta_table = {}
 
    def get_eta(self, runway_id: str, gate_id: str, flight_type: str = "commercial") -> float:
        """Return ETA in minutes for runway→gate given time bucket and flight type."""
        bucket = self.state.get_time_bucket()
        try:
            base = (
                self._eta_table["time_buckets"][bucket]["runway_to_gate"]
                [runway_id][gate_id]
            )
            speed_factor = (
                self._eta_table.get("flight_type_taxi_speed", {}).get(flight_type, 1.0)
            )
            return round(base * speed_factor, 1)
        except (KeyError, TypeError):
            return 8.0  # sensible fallback
 
    def get_optimal_runway_for_gate(self, gate_id: str) -> Optional[str]:
        """Return the runway with the lowest ETA to gate_id that is currently free."""
        bucket = self.state.get_time_bucket()
        best_runway = None
        best_time = float("inf")
        for runway_id in self.state.get_available_runways():
            eta = self.get_eta(runway_id, gate_id)
            if eta < best_time:
                best_time = eta
                best_runway = runway_id
        return best_runway
 
    def score_eta_optimality(self, runway_id: str, gate_id: str, flight_type: str = "commercial") -> float:
        """Score 0.0–1.0: 1.0 if optimal runway chosen, decreases proportionally for slower choices."""
        available = self.state.get_available_runways()
        if not available:
            return 1.0
        chosen_eta = self.get_eta(runway_id, gate_id, flight_type)
        all_etas = [self.get_eta(r, gate_id, flight_type) for r in available]
        min_eta = min(all_etas)
        max_eta = max(all_etas)
        if max_eta == min_eta:
            return 1.0
        # Inverted normalisation: lower ETA = higher score
        return round(1.0 - (chosen_eta - min_eta) / (max_eta - min_eta), 4)
 
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
            elif r.get("status") == "occupied":
                status = RunwayStatus.OCCUPIED
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
            flight = self.state.flights.get(runway.assigned_flight)
            if flight:
                flight.assigned_runway = None
                flight.current_location = None
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
        # Free the runway this flight was on (it's taxiing to gate now)
        flight = self.state.flights[flight_id]
        if flight.assigned_runway:
            self.vacate_runway(flight.assigned_runway)
        gate.status = GateStatus.OCCUPIED
        gate.assigned_flight = flight_id
        flight.assigned_gate = gate_id
        flight.current_location = gate_id
        flight.status = "at_gate"
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
 
    def resolve_crisis(self, flight_id: str) -> bool:
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
                # Fuel drains 1 min per step (each step = ~1 real minute at taxi speed)
                if flight.fuel_remaining_mins > 0:
                    flight.fuel_remaining_mins = max(0, flight.fuel_remaining_mins - 1)
        # Snapshot for grader history (exclude large state_history to avoid recursion)
        snapshot = self.to_dict()
        snapshot.pop("state_history", None)
        self.state.state_history.append(snapshot)
 
    def to_dict(self) -> dict:
        return {
            "step": self.state.step,
            "time_context": self.state.time_context.model_dump(),
            "runways": {k: v.model_dump() for k, v in self.state.runways.items()},
            "gates": {k: v.model_dump() for k, v in self.state.gates.items()},
            "flights": {k: v.model_dump() for k, v in self.state.flights.items()},
            "ground_units": self.state.ground_units.model_dump(),
            "active_crises": [c.model_dump() for c in self.state.active_crises],
            "state_history": self.state.state_history,
        }
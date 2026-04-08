from typing import List, Optional, Literal
from pydantic import BaseModel, Field
 
 
class FlightInfo(BaseModel):
    flight_id: str
    flight_type: Literal["army", "medevac", "government", "commercial", "cargo"]
    status: Literal[
        "requesting_landing", "requesting_takeoff", "at_gate", "holding", "diverted"
    ]
    fuel_remaining_mins: int
    passengers: int
    crisis: Optional[
        Literal["hijack", "bomb_threat", "fire", "medical_onboard"]
    ] = None
 
 
class Observation(BaseModel):
    step: int
    time_of_day: str
    day_of_week: str
    is_holiday: bool
    flights: List[FlightInfo]
    runways: dict
    gates: dict
    active_crises: List[str]
    available_runways: List[str] = Field(default_factory=list)
    available_gates: List[str] = Field(default_factory=list)
    ground_units: dict = Field(default_factory=dict)
 
 
class Action(BaseModel):
    flight_id: str
    action_type: Literal[
        "assign_runway",
        "assign_gate",
        "hold",
        "divert",
        "scramble_security",
        "scramble_fire",
        "scramble_medical",
        "close_runway",
        "vacate_runway",
    ]
    target_id: Optional[str] = None
    use_secure_channel: bool = False
    notify_authorities: Optional[List[Literal["security", "fire", "police"]]] = None
 
 
class Reward(BaseModel):
    total: float
    priority_score: float
    resource_match_score: float
    eta_score: float
    penalty: float
    crisis_protocol_score: float
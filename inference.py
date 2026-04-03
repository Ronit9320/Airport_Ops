import os
import json
import requests
from openai import OpenAI
from typing import Optional


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

client = OpenAI(base_url=API_BASE_URL, api_key=os.environ.get("OPENAI_API_KEY", "x"))


PRIORITY_ORDER = {
    "army": 1,
    "medevac": 2,
    "government": 3,
    "commercial": 4,
    "cargo": 5,
}


SYSTEM_PROMPT = """You are an airport ground operations controller.
You receive airport state and must decide one action per step.
Priority order: army > medevac > government > commercial > cargo.
Fuel emergency (< 10 mins) overrides ALL priorities.
Crisis protocols:
- Medical emergency: clear nearest runway, assign medical gate, scramble ambulance
- Bomb threat: assign isolation bay, halt movement, use isolation gate
- Hijacking: assign remote stand, use_secure_channel=true, scramble security
- Runway fire: issue go-around, scramble fire, close runway
- Fuel emergency: assign closest runway immediately
Always respond with valid JSON matching the Action schema."""


def get_flight_priority(flight: dict) -> tuple[int, bool]:
    fuel = flight.get("fuel_remaining_mins", 999)
    fuel_emergency = fuel < 10

    flight_type = flight.get("flight_type", "commercial")

    if fuel_emergency:
        if flight_type in ["commercial", "cargo"]:
            return 1, True
        elif flight_type == "government":
            return 2, True

    return PRIORITY_ORDER.get(flight_type, 6), fuel_emergency


def select_action(obs: dict, state: dict) -> dict:
    flights = obs.get("flights", [])
    active_crises = obs.get("active_crises", [])
    runways = obs.get("runways", {})
    gates = obs.get("gates", {})

    crisis_flights = {cid: None for cid in active_crises}
    for flight in flights:
        if flight.get("flight_id") in active_crises:
            crisis_flights[flight.get("flight_id")] = flight

    for fid in active_crises:
        if fid in crisis_flights and crisis_flights[fid]:
            flight = crisis_flights[fid]
            crisis_type = flight.get("crisis")

            if crisis_type == "medical_onboard" or crisis_type == "medical_emergency":
                for rid, r in runways.items():
                    if r.get("status") == "free":
                        return {
                            "flight_id": fid,
                            "action_type": "assign_runway",
                            "target_id": rid,
                        }
                for gid, g in gates.items():
                    if g.get("status") == "free" and "medical" in gid.lower():
                        return {
                            "flight_id": fid,
                            "action_type": "assign_gate",
                            "target_id": gid,
                        }
                return {"flight_id": fid, "action_type": "scramble_medical"}

            elif crisis_type == "bomb_threat":
                for gid, g in gates.items():
                    if g.get("status") == "free" and "isolation" in gid.lower():
                        return {
                            "flight_id": fid,
                            "action_type": "assign_gate",
                            "target_id": gid,
                        }
                return {"flight_id": fid, "action_type": "hold"}

            elif crisis_type == "hijack":
                for gid, g in gates.items():
                    if g.get("status") == "free" and "isolation" in gid.lower():
                        return {
                            "flight_id": fid,
                            "action_type": "assign_gate",
                            "target_id": gid,
                            "use_secure_channel": True,
                        }
                return {"flight_id": fid, "action_type": "scramble_security"}

            elif crisis_type == "fire":
                for fid2 in active_crises:
                    if fid2 != fid:
                        for flight in flights:
                            if flight.get("flight_id") == fid2:
                                return {"flight_id": fid2, "action_type": "hold"}
                return {"flight_id": fid, "action_type": "scramble_fire"}

    requesting = [
        f for f in flights if f.get("status") in ["requesting_landing", "holding"]
    ]

    requesting.sort(key=lambda x: get_flight_priority(x))

    for flight in requesting:
        fuel = flight.get("fuel_remaining_mins", 999)
        if fuel < 10:
            for rid, r in runways.items():
                if r.get("status") == "free":
                    return {
                        "flight_id": flight.get("flight_id"),
                        "action_type": "assign_runway",
                        "target_id": rid,
                    }

    for flight in requesting:
        for rid, r in runways.items():
            if r.get("status") == "free":
                return {
                    "flight_id": flight.get("flight_id"),
                    "action_type": "assign_runway",
                    "target_id": rid,
                }

    if requesting:
        return {"flight_id": requesting[0].get("flight_id"), "action_type": "hold"}

    return {
        "flight_id": flights[0].get("flight_id") if flights else "FL001",
        "action_type": "hold",
    }


def run_task(task_id: str) -> dict:
    try:
        resp = requests.post(
            f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30
        )
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "task_id": task_id, "message": str(e)}))
        return {"task_id": task_id, "error": str(e)}

    print(json.dumps({"type": "[START]", "task_id": task_id}))
    done = False
    step_count = 0

    while not done and step_count < 100:
        step_count += 1
        try:
            action = select_action(obs, {})
            result = requests.post(f"{ENV_URL}/step", json=action, timeout=30).json()
            print(
                json.dumps(
                    {
                        "type": "[STEP]",
                        "step": step_count,
                        "action": action,
                        "reward": result.get("reward", {}),
                    }
                )
            )

            obs = result.get("observation", {})
            done = result.get("done", False)
        except Exception as e:
            print(
                json.dumps({"type": "[ERROR]", "step": step_count, "message": str(e)})
            )
            break

    final_reward = (
        result.get("reward", {}).get("total", 0.0) if "result" in dir() else 0.0
    )
    print(
        json.dumps(
            {
                "type": "[END]",
                "task_id": task_id,
                "final_reward": final_reward,
                "steps": step_count,
            }
        )
    )
    return {"task_id": task_id, "final_reward": final_reward, "steps": step_count}


if __name__ == "__main__":
    for task in ["task1", "task2", "task3"]:
        run_task(task)

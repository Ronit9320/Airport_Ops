"""
AirportOpsEnv — Baseline Inference Script
Mandatory for OpenEnv hackathon submission.
Emits [START], [STEP], [END] structured logs to stdout.
Uses OpenAI client with API_BASE_URL + MODEL_NAME from env vars.
"""
import os
import json
import requests
import time
from openai import OpenAI
from typing import Optional
 
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4")
API_KEY: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "x"))
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")
 
MAX_STEPS_PER_TASK = {"task1": 20, "task2": 40, "task3": 80}
SUCCESS_THRESHOLD = 0.6
 
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
 
SYSTEM_PROMPT = """You are an Airport Ground Operations Controller.
Each step you receive the current airport state as JSON and must output exactly ONE action as JSON.
 
PRIORITY ORDER (highest first):
1. Fuel emergency (< 10 mins fuel) — overrides ALL other priorities including army
2. Army / defense flights
3. Medevac / medical flights
4. Government / VIP flights
5. Commercial passenger flights
6. Cargo flights (lowest — never preempt others)
 
CRISIS PROTOCOLS (follow exactly — violations cause zero reward):
- medical_emergency: action_type=assign_runway first, then assign_gate to a medical gate (type=medical), then scramble_medical
- bomb_threat: action_type=assign_gate to isolation gate (type=isolation), then hold nearby flights
- hijacking: action_type=assign_gate to isolation gate, set use_secure_channel=true (MANDATORY), then scramble_security
- runway_fire: action_type=hold for approaching flights, then scramble_fire, then close_runway
- fuel_emergency: action_type=assign_runway to ANY free runway IMMEDIATELY (one step delay = hard penalty)
 
HARD RULES (violation = total reward 0.0):
- NEVER assign a hijacked or bomb-threat flight to a pax gate
- NEVER use a runway with status=maintenance or status=closed
- ALWAYS set use_secure_channel=true for hijacking protocol
 
OUTPUT FORMAT — respond with only valid JSON, no markdown, no explanation:
{
  "flight_id": "FL001",
  "action_type": "assign_runway",
  "target_id": "R1",
  "use_secure_channel": false
}
 
Available action_types: assign_runway, assign_gate, hold, divert, scramble_security, scramble_fire, scramble_medical, close_runway, vacate_runway
"""
 
 
def log_start(task_id: str, model: str) -> None:
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "model": model,
        "env_url": ENV_URL,
    }), flush=True)
 
 
def log_step(step: int, action: dict, reward: dict, done: bool, error: Optional[str] = None) -> None:
    print(json.dumps({
        "type": "[STEP]",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error,
    }), flush=True)
 
 
def log_end(task_id: str, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "success": success,
        "steps": steps,
        "score": round(score, 4),
        "rewards": [round(r, 4) for r in rewards],
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
    }), flush=True)
 
 
PRIORITY_MAP = {"army": 1, "medevac": 2, "government": 3, "commercial": 4, "cargo": 5}
 
 
def get_flight_priority(flight: dict) -> int:
    if flight.get("fuel_remaining_mins", 999) < 10:
        return 0
    return PRIORITY_MAP.get(flight.get("flight_type", "commercial"), 6)
 
 
def heuristic_action(obs: dict) -> dict:
    """
    Deterministic fallback heuristic agent — used when LLM call fails.
    Follows exact protocol order for all crisis types.
    """
    flights = obs.get("flights", [])
    active_crises = obs.get("active_crises", [])
    runways = obs.get("runways", {})
    gates = obs.get("gates", {})
    available_runways = obs.get("available_runways", [])
    available_gates = obs.get("available_gates", [])
 
    # Build lookup maps
    flight_map = {f["flight_id"]: f for f in flights}
    free_runways = [rid for rid, r in runways.items() if r.get("status") == "free"]
    free_gates_by_type = {}
    for gid, g in gates.items():
        if g.get("status") == "free":
            gtype = g.get("type", "pax")
            free_gates_by_type.setdefault(gtype, []).append(gid)
 
    # --- Crisis handling first ---
    for fid in active_crises:
        flight = flight_map.get(fid)
        if not flight:
            continue
        crisis = flight.get("crisis")
 
        if crisis == "medical_onboard":
            # Step 1: runway
            if flight.get("status") in ("requesting_landing", "holding") and free_runways:
                return {"flight_id": fid, "action_type": "assign_runway", "target_id": free_runways[0], "use_secure_channel": False}
            # Step 2: medical gate
            if flight.get("assigned_runway") and free_gates_by_type.get("medical"):
                return {"flight_id": fid, "action_type": "assign_gate", "target_id": free_gates_by_type["medical"][0], "use_secure_channel": False}
            # Step 3: ambulance
            return {"flight_id": fid, "action_type": "scramble_medical", "use_secure_channel": False}
 
        elif crisis == "hijack":
            # All three steps in sequence
            if free_gates_by_type.get("isolation"):
                return {"flight_id": fid, "action_type": "assign_gate", "target_id": free_gates_by_type["isolation"][0], "use_secure_channel": True}
            return {"flight_id": fid, "action_type": "scramble_security", "use_secure_channel": False}
 
        elif crisis == "bomb_threat":
            if free_gates_by_type.get("isolation"):
                return {"flight_id": fid, "action_type": "assign_gate", "target_id": free_gates_by_type["isolation"][0], "use_secure_channel": False}
            return {"flight_id": fid, "action_type": "hold", "use_secure_channel": False}
 
        elif crisis == "fire":
            # Hold other approaching flights first
            for other in flights:
                if other["flight_id"] != fid and other.get("status") == "requesting_landing":
                    return {"flight_id": other["flight_id"], "action_type": "hold", "use_secure_channel": False}
            return {"flight_id": fid, "action_type": "scramble_fire", "use_secure_channel": False}
 
    # --- Fuel emergencies next ---
    requesting = [f for f in flights if f.get("status") in ("requesting_landing", "holding")]
    requesting.sort(key=get_flight_priority)
 
    for flight in requesting:
        if flight.get("fuel_remaining_mins", 999) < 10 and free_runways:
            return {"flight_id": flight["flight_id"], "action_type": "assign_runway", "target_id": free_runways[0], "use_secure_channel": False}
 
    # --- Normal priority dispatch ---
    for flight in requesting:
        fid = flight["flight_id"]
        ftype = flight.get("flight_type", "commercial")
 
        # Assign runway if waiting to land
        if flight.get("status") == "requesting_landing" and free_runways:
            return {"flight_id": fid, "action_type": "assign_runway", "target_id": free_runways[0], "use_secure_channel": False}
 
        # Assign gate if already on runway
        if flight.get("assigned_runway"):
            if ftype == "cargo":
                gate_type = "cargo"
            elif ftype == "medevac":
                gate_type = "medical"
            else:
                gate_type = "pax"
            if free_gates_by_type.get(gate_type):
                return {"flight_id": fid, "action_type": "assign_gate", "target_id": free_gates_by_type[gate_type][0], "use_secure_channel": False}
 
    # Hold highest priority flight if no resources
    if requesting:
        return {"flight_id": requesting[0]["flight_id"], "action_type": "hold", "use_secure_channel": False}
 
    # Fallback — hold first flight
    if flights:
        return {"flight_id": flights[0]["flight_id"], "action_type": "hold", "use_secure_channel": False}
 
    return {"flight_id": "FL001", "action_type": "hold", "use_secure_channel": False}
 
 
def get_llm_action(obs: dict, history: list[dict]) -> dict:
    """Call the LLM to get the next action. Falls back to heuristic on failure."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
    # Include last 3 steps of history for context
    for h in history[-3:]:
        messages.append({"role": "user", "content": json.dumps(h["obs"])})
        messages.append({"role": "assistant", "content": json.dumps(h["action"])})
 
    messages.append({
        "role": "user",
        "content": f"Current airport state:\n{json.dumps(obs, indent=2)}\n\nWhat is your action?"
    })
 
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if model returns them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw)
        # Ensure required fields exist
        action.setdefault("use_secure_channel", False)
        return action
    except Exception as exc:
        print(json.dumps({"type": "[DEBUG]", "msg": f"LLM failed: {exc}, using heuristic"}), flush=True)
        return heuristic_action(obs)
 
 
def run_task(task_id: str) -> dict:
    max_steps = MAX_STEPS_PER_TASK.get(task_id, 20)
 
    # Reset environment
    try:
        resp = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(json.dumps({"type": "[ERROR]", "task_id": task_id, "message": str(e)}), flush=True)
        return {"task_id": task_id, "error": str(e), "score": 0.0}
 
    log_start(task_id, MODEL_NAME)
 
    done = False
    step_count = 0
    rewards: list[float] = []
    history: list[dict] = []
    last_result = {}
 
    while not done and step_count < max_steps:
        step_count += 1
 
        action = get_llm_action(obs, history)
 
        try:
            result = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
            result.raise_for_status()
            last_result = result.json()
        except Exception as e:
            log_step(step_count, action, {}, False, error=str(e))
            break
 
        reward_obj = last_result.get("reward", {})
        reward_val = reward_obj.get("total", 0.0) if isinstance(reward_obj, dict) else 0.0
        done = last_result.get("done", False)
 
        rewards.append(reward_val)
        history.append({"obs": obs, "action": action})
        obs = last_result.get("observation", obs)
 
        log_step(step_count, action, reward_obj, done)
 
        if done:
            break
 
    score = sum(rewards) / len(rewards) if rewards else 0.0
    score = round(min(max(score, 0.0), 1.0), 4)
    success = score >= SUCCESS_THRESHOLD
 
    log_end(task_id, success, step_count, score, rewards)
    return {"task_id": task_id, "score": score, "steps": step_count, "success": success}
 
 
if __name__ == "__main__":
    print(json.dumps({"type": "[INFO]", "msg": f"Starting AirportOpsEnv baseline — model: {MODEL_NAME}"}), flush=True)
    results = []
    for task in ["task1", "task2", "task3"]:
        result = run_task(task)
        results.append(result)
        time.sleep(1)
 
    print(json.dumps({
        "type": "[SUMMARY]",
        "results": results,
        "overall_score": round(sum(r.get("score", 0) for r in results) / len(results), 4),
    }), flush=True)
 
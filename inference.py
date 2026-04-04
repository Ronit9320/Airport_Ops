"""
AirportOpsEnv — Baseline Inference Script
Mandatory file name: inference.py in project root.
Uses OpenAI client. Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars.
Emits structured [START], [STEP], [END] logs to stdout.
"""
import os
import json
import time
import requests
from openai import OpenAI
from typing import Optional
 
# ── Config ──────────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4")
API_KEY: str = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", "x"))
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")
 
MAX_STEPS: dict[str, int] = {"task1": 20, "task2": 40, "task3": 80}
SUCCESS_THRESHOLD = 0.6
 
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
 
# ── Priority helper (single definition — Fix #7) ─────────────────────
_PRIORITY_MAP = {"army": 1, "medevac": 2, "government": 3, "commercial": 4, "cargo": 5}
 
 
def get_flight_priority(flight: dict) -> int:
    """Lower = higher priority. Fuel emergency (< 10 mins) overrides all."""
    if flight.get("fuel_remaining_mins", 999) < 10:
        return 0
    return _PRIORITY_MAP.get(flight.get("flight_type", "commercial"), 6)
 
 
# ── Logging ──────────────────────────────────────────────────────────────
 
def log_start(task_id: str, model: str) -> None:
    print(json.dumps({"type": "[START]", "task_id": task_id, "model": model}), flush=True)
 
 
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
 
 
# ── System prompt ────────────────────────────────────────────────────────
 
SYSTEM_PROMPT = """You are an Airport Ground Operations Controller.
Output ONE action per step as valid JSON only — no markdown, no explanation.
 
PRIORITY ORDER (highest first, lower number = higher priority):
0. Fuel emergency: any flight with fuel_remaining_mins < 10 — assign runway IMMEDIATELY
1. Army / defense
2. Medevac / medical
3. Government / VIP
4. Commercial
5. Cargo (never preempts others)
 
CRISIS PROTOCOLS — follow exactly or receive zero reward:
- medical_onboard: assign_runway → assign_gate (type=medical, id=G_MED) → scramble_medical
- hijack: assign_gate to G_ISO + set use_secure_channel=true (MANDATORY) → scramble_security
- bomb_threat: assign_gate to G_ISO → hold nearby flights
- fire (runway_fire): hold approaching flights → scramble_fire → close_runway
- fuel_emergency: assign_runway to ANY free runway in the SAME step as crisis activation
 
HARD RULES (violation = reward 0.0 for full episode):
- NEVER assign hijack or bomb_threat flight to a pax gate
- NEVER use a runway with status=maintenance or status=closed
- ALWAYS set use_secure_channel=true on hijack gate assignment
 
OUTPUT FORMAT:
{"flight_id": "FL001", "action_type": "assign_runway", "target_id": "R1", "use_secure_channel": false}
 
action_type options: assign_runway, assign_gate, hold, divert,
                     scramble_security, scramble_fire, scramble_medical,
                     close_runway, vacate_runway
"""
 
 
# ── Heuristic fallback agent ─────────────────────────────────────────────
 
def heuristic_action(obs: dict) -> dict:
    """Deterministic fallback used when LLM call fails."""
    flights = obs.get("flights", [])
    active_crises = obs.get("active_crises", [])
    runways = obs.get("runways", {})
    gates = obs.get("gates", {})
 
    flight_map = {f["flight_id"]: f for f in flights}
    free_runways = [rid for rid, r in runways.items() if r.get("status") == "free"]
    free_by_type: dict[str, list[str]] = {}
    for gid, g in gates.items():
        if g.get("status") == "free":
            free_by_type.setdefault(g.get("type", "pax"), []).append(gid)
 
    # Crisis handling first
    for fid in active_crises:
        flight = flight_map.get(fid)
        if not flight:
            continue
        crisis = flight.get("crisis")
 
        if crisis == "medical_onboard":
            if flight.get("status") in ("requesting_landing", "holding") and free_runways:
                return {"flight_id": fid, "action_type": "assign_runway",
                        "target_id": free_runways[0], "use_secure_channel": False}
            if free_by_type.get("medical"):
                return {"flight_id": fid, "action_type": "assign_gate",
                        "target_id": free_by_type["medical"][0], "use_secure_channel": False}
            return {"flight_id": fid, "action_type": "scramble_medical", "use_secure_channel": False}
 
        elif crisis == "hijack":
            if free_by_type.get("isolation"):
                return {"flight_id": fid, "action_type": "assign_gate",
                        "target_id": free_by_type["isolation"][0], "use_secure_channel": True}
            return {"flight_id": fid, "action_type": "scramble_security", "use_secure_channel": False}
 
        elif crisis == "bomb_threat":
            if free_by_type.get("isolation"):
                return {"flight_id": fid, "action_type": "assign_gate",
                        "target_id": free_by_type["isolation"][0], "use_secure_channel": False}
            return {"flight_id": fid, "action_type": "hold", "use_secure_channel": False}
 
        elif crisis == "fire":
            for other in flights:
                if other["flight_id"] != fid and other.get("status") == "requesting_landing":
                    return {"flight_id": other["flight_id"], "action_type": "hold",
                            "use_secure_channel": False}
            return {"flight_id": fid, "action_type": "scramble_fire", "use_secure_channel": False}
 
    # Fuel emergencies
    requesting = sorted(
        [f for f in flights if f.get("status") in ("requesting_landing", "holding")],
        key=get_flight_priority
    )
    for flight in requesting:
        if flight.get("fuel_remaining_mins", 999) < 10 and free_runways:
            return {"flight_id": flight["flight_id"], "action_type": "assign_runway",
                    "target_id": free_runways[0], "use_secure_channel": False}
 
    # Normal dispatch
    for flight in requesting:
        fid = flight["flight_id"]
        ftype = flight.get("flight_type", "commercial")
        if flight.get("status") == "requesting_landing" and free_runways:
            return {"flight_id": fid, "action_type": "assign_runway",
                    "target_id": free_runways[0], "use_secure_channel": False}
        if flight.get("assigned_runway"):
            gate_type = "medical" if ftype == "medevac" else ("cargo" if ftype == "cargo" else "pax")
            if free_by_type.get(gate_type):
                return {"flight_id": fid, "action_type": "assign_gate",
                        "target_id": free_by_type[gate_type][0], "use_secure_channel": False}
 
    if requesting:
        return {"flight_id": requesting[0]["flight_id"], "action_type": "hold",
                "use_secure_channel": False}
    if flights:
        return {"flight_id": flights[0]["flight_id"], "action_type": "hold",
                "use_secure_channel": False}
    return {"flight_id": "FL001", "action_type": "hold", "use_secure_channel": False}
 
 
# ── LLM agent ────────────────────────────────────────────────────────────
 
def get_llm_action(obs: dict, history: list[dict]) -> dict:
    """Call MODEL_NAME via OpenAI client. Falls back to heuristic on any failure."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-3:]:
        messages.append({"role": "user", "content": json.dumps(h["obs"])})
        messages.append({"role": "assistant", "content": json.dumps(h["action"])})
    messages.append({
        "role": "user",
        "content": f"Current state:\n{json.dumps(obs, indent=2)}\n\nWhat is your action?"
    })
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=200,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        action = json.loads(raw)
        action.setdefault("use_secure_channel", False)
        return action
    except Exception as exc:
        print(json.dumps({"type": "[DEBUG]", "msg": f"LLM failed: {exc}, using heuristic"}),
              flush=True)
        return heuristic_action(obs)
 
 
# ── Task runner ──────────────────────────────────────────────────────────
 
def run_task(task_id: str) -> dict:
    max_steps = MAX_STEPS.get(task_id, 20)
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
    last_result: dict = {}
 
    while not done and step_count < max_steps:
        step_count += 1
        action = get_llm_action(obs, history)
        try:
            r = requests.post(f"{ENV_URL}/step", json=action, timeout=30)
            r.raise_for_status()
            last_result = r.json()
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
 
    score = round(min(max(sum(rewards) / len(rewards) if rewards else 0.0, 0.0), 1.0), 4)
    success = score >= SUCCESS_THRESHOLD
    log_end(task_id, success, step_count, score, rewards)
    return {"task_id": task_id, "score": score, "steps": step_count, "success": success}
 
 
# ── Entry point ──────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print(json.dumps({"type": "[INFO]", "msg": f"AirportOpsEnv baseline — model: {MODEL_NAME}"}),
          flush=True)
    results = []
    for task in ["task1", "task2", "task3"]:
        result = run_task(task)
        results.append(result)
        time.sleep(1)
 
    overall = round(sum(r.get("score", 0.0) for r in results) / len(results), 4)
    print(json.dumps({"type": "[SUMMARY]", "results": results, "overall_score": overall}),
          flush=True)
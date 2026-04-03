from fastapi import FastAPI
from models import Observation, Action, Reward
from env import AirportOpsEnv

app = FastAPI()
env = AirportOpsEnv()


@app.post("/reset")
def reset(task_id: str = "task1") -> Observation:
    return env.reset(task_id)


@app.post("/step")
def step(action: Action) -> dict:
    obs, reward, done, info = env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state() -> dict:
    return env.state()


@app.get("/health")
def health():
    return {"status": "ok"}

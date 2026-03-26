
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from environment import DeliveryDecisionEnvironment, TASKS

app = FastAPI(title="Smart Delivery RL Environment")

# Global env instances per task
envs = {task_id: DeliveryDecisionEnvironment(task_id=task_id) for task_id in TASKS}
current_env = envs["task_1"]

class ActionRequest(BaseModel):
    action_type: str
    task_id: Optional[str] = "task_1"

@app.get("/")
def root():
    return {"name": "Smart Delivery Env", "version": "1.0.0",
            "endpoints": ["/tasks", "/reset", "/step", "/state", "/grader", "/baseline"]}

@app.get("/tasks")
def get_tasks():
    return {"tasks": [
        {"task_id": k, "name": v["name"],
         "difficulty": v["difficulty"],
         "description": v["description"]}
        for k, v in TASKS.items()
    ]}

@app.post("/reset")
def reset(task_id: str = "task_1"):
    global current_env
    if task_id not in envs:
        return {"error": f"Unknown task_id. Choose: {list(TASKS.keys())}"}
    current_env = envs[task_id]
    obs = current_env.reset()
    return obs.dict()

@app.post("/step")
def step(req: ActionRequest):
    global current_env
    if req.task_id and req.task_id in envs:
        current_env = envs[req.task_id]
    obs = current_env.step(req.action_type)
    return obs.dict()

@app.get("/state")
def state():
    return current_env.state.dict()

@app.post("/grader")
def grader(task_id: str = "task_1"):
    if task_id not in envs:
        return {"error": "Unknown task_id"}
    score = envs[task_id].grade()
    return {"task_id": task_id, "score": score,
            "outcome": envs[task_id].state.outcome,
            "steps": envs[task_id].state.step_count}

@app.post("/baseline")
def baseline(task_id: str = "task_1"):
    """Rule-based baseline agent — no OpenAI key needed."""
    if task_id not in TASKS:
        return {"error": "Unknown task_id"}
    env = DeliveryDecisionEnvironment(task_id=task_id)
    obs = env.reset()
    trajectory = []
    while not obs.done:
        # Simple rule-based policy
        if obs.locker_nearby > 0.5:
            action = "locker_drop"
        elif obs.alternate_available > 0.5:
            action = "neighbor_handoff"
        elif (obs.location_safety > 0.5 and obs.weather_risk < 0.6
              and obs.package_fragile < 0.5):
            action = "safe_drop"
        elif obs.otp_attempts < 0.7:
            action = "retry"
        elif obs.recipient_available < 0.5:
            action = "contact_support"
        else:
            action = "return"
        trajectory.append({"action": action, "message": obs.message})
        obs = env.step(action)
        if env.state.step_count >= 10:
            break
    return {
        "task_id": task_id,
        "trajectory": trajectory,
        "final_score": env.grade(),
        "outcome": env.state.outcome,
        "total_steps": env.state.step_count
    }

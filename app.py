import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
from models import DeliveryObservation, DeliveryAction
from environment import DeliveryDecisionEnvironment, TASKS

app  = FastAPI(
    title="🚚 Smart Delivery RL Environment",
    description="""
## Meta x HuggingFace x Scaler Hackathon — OpenEnv Track

AI agent learns optimal last-mile delivery decisions.

### 3 Tasks
| Task | Difficulty | Description |
|------|-----------|-------------|
| task_1 | Easy | Low-value, locker nearby |
| task_2 | Medium | Fragile, bad weather |
| task_3 | Hard | High-value, unsafe, time critical |

### Valid Actions
`retry` `safe_drop` `locker_drop` `neighbor_handoff` `contact_support` `return`

**Author:** Ravi Gohel | [GitHub](https://github.com/ravigohel142996/RL-SmartDelivery-Agent)
""",
    version="1.0.0"
)
envs = {t: DeliveryDecisionEnvironment(task_id=t) for t in TASKS}
for t in TASKS: envs[t].reset()

class StepRequest(BaseModel):
    action_type: str
    task_id: Optional[str] = "task_1"

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status":"healthy","environment":"smart-delivery-env",
            "version":"1.0.0","tasks":list(TASKS.keys())}

@app.get("/tasks")
def get_tasks():
    return {"tasks":[{"task_id":k,"name":v["name"],
        "difficulty":v["difficulty"],"description":v["description"],
        "action_schema":{"action_type":"string",
            "valid_values":["retry","safe_drop","locker_drop",
                           "neighbor_handoff","contact_support","return"]}
        } for k,v in TASKS.items()]}

@app.post("/reset")
def reset(task_id: str = "task_1"):
    if task_id not in envs:
        return {"error":f"Unknown task. Choose: {list(TASKS.keys())}"}
    return envs[task_id].reset().dict()

@app.post("/step")
def step(req: StepRequest):
    task_id = req.task_id or "task_1"
    if task_id not in envs: return {"error":"Unknown task_id"}
    return envs[task_id].step(req.action_type).dict()

@app.get("/state")
def state(task_id: str = "task_1"):
    if task_id not in envs: return {"error":"Unknown task_id"}
    return envs[task_id].state.dict()

@app.post("/grader")
def grader(task_id: str = "task_1"):
    if task_id not in envs: return {"error":"Unknown task_id"}
    return {"task_id":task_id,"score":envs[task_id].grade(),
            "outcome":envs[task_id].state.outcome,
            "steps":envs[task_id].state.step_count}

@app.post("/baseline")
def baseline(task_id: str = "task_1"):
    if task_id not in TASKS: return {"error":"Unknown task_id"}
    env = DeliveryDecisionEnvironment(task_id=task_id)
    obs = env.reset(); traj=[]
    while not obs.done and env.state.step_count<10:
        if obs.locker_nearby>0.5:         act="locker_drop"
        elif obs.alternate_available>0.5: act="neighbor_handoff"
        elif (obs.location_safety>0.5 and obs.weather_risk<0.6
              and obs.package_fragile<0.5): act="safe_drop"
        elif obs.otp_attempts<0.7:        act="retry"
        elif obs.recipient_available<0.5: act="contact_support"
        else:                             act="return"
        traj.append({"action":act,"message":obs.message})
        obs=env.step(act)
    return {"task_id":task_id,"trajectory":traj,
            "final_score":env.grade(),"outcome":env.state.outcome,
            "total_steps":env.state.step_count}
